"""Unit tests for intent routing, token budgeting, and agent controls."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services import agent_orchestration as orch
from app import chatbot


class TestIntentRouting(unittest.TestCase):
    def test_greeting_uses_no_tools(self):
        intent = orch.classify_intent("Hi, how are you?")
        self.assertEqual(intent, orch.INTENT_GENERAL)
        self.assertFalse(orch.intent_allows_tools(intent))
        self.assertEqual(orch.tools_for_intent(intent, chatbot.tool_definitions), [])

    def test_rag_is_direct_technical(self):
        intent = orch.classify_intent("What is RAG?")
        self.assertEqual(intent, orch.INTENT_TECHNICAL)
        self.assertFalse(orch.intent_allows_tools(intent))

    def test_codebase_investigation_tools(self):
        intent = orch.classify_intent("Find my agent loop implementation")
        self.assertEqual(intent, orch.INTENT_CODEBASE)
        names = {
            (t.get("function") or {}).get("name")
            for t in orch.tools_for_intent(intent, chatbot.tool_definitions)
        }
        self.assertEqual(names, {"search_code", "read_file"})

    def test_debug_intent(self):
        intent = orch.classify_intent("Investigate my tool execution error")
        self.assertEqual(intent, orch.INTENT_DEBUG)

    def test_calculation_intent(self):
        intent = orch.classify_intent("Calculate 25 * 40")
        self.assertEqual(intent, orch.INTENT_CALC)

    def test_educational_bug_question_no_tools(self):
        """Bug/investigate/fix keywords alone must not activate tools."""
        msg = (
            "Explain how an AI agent can identify, investigate, and fix "
            "a software bug with a real-world example"
        )
        intent = orch.classify_intent(msg)
        self.assertEqual(intent, orch.INTENT_TECHNICAL)
        self.assertFalse(orch.intent_allows_tools(intent))
        self.assertEqual(
            orch.tools_for_intent(intent, chatbot.tool_definitions),
            [],
        )

    def test_conceptual_debugging_how_question_no_tools(self):
        intent = orch.classify_intent(
            "How can an agent investigate and fix bugs in software?"
        )
        self.assertEqual(intent, orch.INTENT_TECHNICAL)
        self.assertFalse(orch.intent_allows_tools(intent))

    def test_example_request_with_bug_keyword_no_tools(self):
        intent = orch.classify_intent(
            "Give me an example of debugging a flaky test failure"
        )
        self.assertEqual(intent, orch.INTENT_TECHNICAL)
        self.assertFalse(orch.intent_allows_tools(intent))

    def test_actual_repo_debug_activates_tools(self):
        intent = orch.classify_intent(
            "Investigate this error in my backend and fix the bug"
        )
        self.assertEqual(intent, orch.INTENT_DEBUG)
        self.assertTrue(orch.intent_allows_tools(intent))
        names = {
            (t.get("function") or {}).get("name")
            for t in orch.tools_for_intent(intent, chatbot.tool_definitions)
        }
        self.assertIn("search_code", names)
        self.assertIn("read_file", names)

    def test_stack_trace_debug_request(self):
        intent = orch.classify_intent(
            "Here is my stack trace — diagnose this exception in the agent loop"
        )
        self.assertEqual(intent, orch.INTENT_DEBUG)

    def test_repo_path_investigation_not_educational(self):
        intent = orch.classify_intent(
            "Read app/chatbot.py and find why my tool calls are failing"
        )
        self.assertIn(intent, {orch.INTENT_DEBUG, orch.INTENT_CODEBASE})
        self.assertTrue(orch.intent_allows_tools(intent))

    def test_memory_and_current_info_preserved(self):
        self.assertEqual(
            orch.classify_intent("Remember that I prefer dark mode"),
            orch.INTENT_MEMORY,
        )
        self.assertEqual(
            orch.classify_intent("What is the current time?"),
            orch.INTENT_CURRENT,
        )

    def test_natural_preference_recall_routes_to_personal_memory(self):
        for message in (
            "retrieve my saved preferences",
            "retrieve my previously saved preferences",
            "what are my preferences",
            "what preferences do I have?",
            "show my preferences",
            "which programming language do I prefer?",
        ):
            with self.subTest(message=message):
                intent = orch.classify_intent(message)
                self.assertEqual(intent, orch.INTENT_MEMORY)
                self.assertTrue(orch.intent_needs_long_term_memory(intent, message))

        # Personal recall routing must not capture repository inspection.
        self.assertEqual(
            orch.classify_intent("Find my agent loop implementation"),
            orch.INTENT_CODEBASE,
        )


class TestAuthenticatedMemoryRetrieval(unittest.TestCase):
    def test_memory_search_uses_each_authenticated_identity_and_tenant(self):
        results = {
            "usr_account_a": [{"memory": "A preference", "status": "active"}],
            "usr_account_b": [{"memory": "B preference", "status": "active"}],
        }

        def search(user_id, query, **kwargs):
            return {"success": True, "memories": results[user_id], "conflicts": [], "summaries": []}

        with patch.object(chatbot, "semantic_search_memories", side_effect=search) as mocked:
            account_a = chatbot.load_long_term_memory(
                "usr_account_a", "What are my preferences?", tenant_id="ten_a"
            )
            account_b = chatbot.load_long_term_memory(
                "usr_account_b", "What are my preferences?", tenant_id="ten_b"
            )

        self.assertEqual(account_a["memories"][0]["memory"], "A preference")
        self.assertEqual(account_b["memories"][0]["memory"], "B preference")
        self.assertEqual(mocked.call_args_list[0].kwargs["user_id"], "usr_account_a")
        self.assertEqual(mocked.call_args_list[0].kwargs["tenant_id"], "ten_a")
        self.assertEqual(mocked.call_args_list[1].kwargs["user_id"], "usr_account_b")
        self.assertEqual(mocked.call_args_list[1].kwargs["tenant_id"], "ten_b")

    def test_conceptual_agent_memory_topics_are_technical_no_tools(self):
        """Agent LTM concepts stay tool-free; not personal-memory lookup."""
        msg = (
            "Explain how AI agent memory scoping, conflict resolution, "
            "freshness, reliability, summarization, reinforcement, "
            "source traceability and deletion work."
        )
        intent = orch.classify_intent(msg)
        self.assertEqual(intent, orch.INTENT_TECHNICAL)
        self.assertFalse(orch.intent_allows_tools(intent))
        self.assertEqual(
            orch.tools_for_intent(intent, chatbot.tool_definitions),
            [],
        )
        self.assertFalse(chatbot.is_memory_metadata_inspect_request(msg))
        self.assertIsNone(chatbot.try_inspect_memory_metadata(msg, "u1"))
        prompt = orch.DIRECT_ANSWER_SYSTEM_PROMPT.lower()
        self.assertIn("agent long-term memory", prompt)
        self.assertIn("scoping", prompt)
        self.assertIn("never as programming-language variable scope", prompt)

    def test_memory_architecture_inspection_is_codebase_not_metadata(self):
        """Repo memory-service inspection must not ask for a fact_key."""
        msg = (
            "Inspect my memory architecture. Verify implementations, "
            "read complete implementations and relevant tests. Do not modify."
        )
        intent = orch.classify_intent(msg)
        self.assertEqual(intent, orch.INTENT_CODEBASE)
        self.assertTrue(orch.intent_allows_tools(intent))
        names = {
            (t.get("function") or {}).get("name")
            for t in orch.tools_for_intent(intent, chatbot.tool_definitions)
        }
        self.assertEqual(names, {"search_code", "read_file"})
        self.assertTrue(chatbot.is_memory_system_code_inspection_request(msg))
        self.assertFalse(chatbot.is_memory_metadata_inspect_request(msg))
        self.assertIsNone(chatbot.try_inspect_memory_metadata(msg, "u1"))

    def test_memory_permissions_implementation_is_codebase_investigation(self):
        """Topic 21 source inspection must not short-circuit to fact_key metadata."""
        msg = (
            "Inspect my memory permissions implementation. Verify ownership, "
            "read/write access, shared memory, unauthorized access prevention, "
            "and permission-aware semantic retrieval. Read implementations "
            "and tests. Report PASS/PARTIAL/FAIL with exact file paths, line "
            "numbers, and evidence. Do not modify files."
        )
        intent = orch.classify_intent(msg)
        self.assertEqual(intent, orch.INTENT_CODEBASE)
        self.assertTrue(chatbot.is_memory_system_code_inspection_request(msg))
        self.assertFalse(chatbot.is_memory_metadata_inspect_request(msg))
        self.assertIsNone(chatbot.try_inspect_memory_metadata(msg, "u1"))
        names = {
            (t.get("function") or {}).get("name")
            for t in orch.tools_for_intent(intent, chatbot.tool_definitions)
        }
        self.assertEqual(names, {"search_code", "read_file"})

    def test_memory_scoping_source_code_is_codebase_investigation(self):
        msg = "Inspect my memory scoping source code"
        intent = orch.classify_intent(msg)
        self.assertEqual(intent, orch.INTENT_CODEBASE)
        self.assertTrue(chatbot.is_memory_system_code_inspection_request(msg))
        self.assertFalse(chatbot.is_memory_metadata_inspect_request(msg))

    def test_retrieve_preferred_database_memory_is_memory_operation(self):
        intent = orch.classify_intent(
            "Retrieve my preferred database memory"
        )
        self.assertEqual(intent, orch.INTENT_MEMORY)
        self.assertFalse(
            chatbot.is_memory_system_code_inspection_request(
                "Retrieve my preferred database memory"
            )
        )

    def test_delete_stored_memory_is_memory_operation(self):
        intent = orch.classify_intent("Delete a stored memory")
        self.assertEqual(intent, orch.INTENT_MEMORY)


class TestHistoryCompaction(unittest.TestCase):
    def test_compacts_large_history_before_groq(self):
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"question {i} " + ("x" * 40)})
            history.append({"role": "assistant", "content": f"answer {i} " + ("y" * 40)})
        compacted = orch.compact_conversation_history(history, max_messages=6)
        self.assertLess(len(compacted), len(history))
        self.assertTrue(
            any(
                m.get("role") == "system"
                and "compacted" in (m.get("content") or "").lower()
                for m in compacted
            )
        )

    def test_preserves_tool_call_sequences(self):
        history = [
            {"role": "user", "content": "find loop"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "search_code", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "name": "search_code", "content": "hit"},
            {"role": "assistant", "content": "found it"},
        ]
        # Pad with older turns
        older = []
        for i in range(10):
            older.append({"role": "user", "content": f"old {i}"})
            older.append({"role": "assistant", "content": f"old ans {i}"})
        compacted = orch.compact_conversation_history(
            older + history, max_messages=4
        )
        # Tool result should still follow its assistant tool_calls if kept.
        for i, msg in enumerate(compacted):
            if msg.get("role") == "tool":
                self.assertGreater(i, 0)
                self.assertEqual(compacted[i - 1].get("role"), "assistant")
                self.assertTrue(compacted[i - 1].get("tool_calls"))


class TestTokenBudgetAnd413(unittest.TestCase):
    def test_prune_reduces_oversized_payload(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 20000},
            {"role": "tool", "content": "y" * 15000, "tool_call_id": "1", "name": "read_file"},
        ]
        pruned = orch.prune_messages_for_budget(messages, tool_defs=[], budget=3000)
        total = orch.estimate_messages_tokens(pruned)
        self.assertLess(total, 3000)

    def test_413_triggers_compact_retry(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello! I'm doing well.", tool_calls=None)
            )
        ]
        mock_response.usage = MagicMock(total_tokens=200)

        def _create(**kwargs):
            if not hasattr(_create, "calls"):
                _create.calls = 0
            _create.calls += 1
            if _create.calls == 1:
                raise Exception(
                    "Error code: 413 - Request too large for model. "
                    "requested 8300"
                )
            self.assertNotIn("tools", kwargs)
            self.assertNotIn("tool_choice", kwargs)
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _create

        with patch.object(chatbot, "client", mock_client), patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            # Seed oversized history
            store = chatbot._get_conversation_store("orch-413")
            store.clear()
            for i in range(12):
                store.append({"role": "user", "content": "blob " + ("z" * 500)})
                store.append({"role": "assistant", "content": "ok " + ("z" * 500)})

            answer = chatbot.chat_with_ai(
                "Hi, how are you?",
                user_id="orch-413",
                request_id="r-413",
            )

        self.assertGreaterEqual(_create.calls, 2)
        self.assertIn("well", answer.lower())

    def test_429_tpm_is_not_classified_as_413(self):
        err = Exception(
            "Error code: 429 - Rate limit reached for model on tokens per "
            "minute (TPM): Limit 8000, Used 3433, Requested 5174. "
            "Please try again in 4.5525s."
        )
        self.assertEqual(orch.classify_groq_rate_limit(err), "tpm")
        self.assertTrue(orch.is_tpm_rate_limit_error(err))
        self.assertFalse(orch.is_tpd_rate_limit_error(err))
        self.assertTrue(orch.is_rate_limit_error(err))
        self.assertFalse(orch.is_token_limit_error(err))
        self.assertAlmostEqual(
            orch.extract_retry_after_seconds(err), 4.5525 + 0.35, places=2
        )

    def test_429_tpd_not_classified_as_tpm(self):
        err = Exception(
            "Error code: 429 - Rate limit reached for model on tokens per "
            "day (TPD): Limit 200000, Used 198569, Requested 4963. "
            "Please try again in 25m25.823999999s."
        )
        self.assertEqual(orch.classify_groq_rate_limit(err), "tpd")
        self.assertTrue(orch.is_tpd_rate_limit_error(err))
        self.assertTrue(orch.is_daily_quota_exhausted(err))
        self.assertFalse(orch.is_tpm_rate_limit_error(err))
        self.assertFalse(orch.is_token_limit_error(err))

    def test_429_rpm_rpd_classified_separately(self):
        rpm = Exception(
            "Error code: 429 - Rate limit reached for model on requests per "
            "minute (RPM): Limit 30, Used 30, Requested 1. "
            "Please try again in 1.2s."
        )
        rpd = Exception(
            "Error code: 429 - Rate limit reached for model on requests per "
            "day (RPD): Limit 14400, Used 14400, Requested 1. "
            "Please try again in 12h."
        )
        self.assertEqual(orch.classify_groq_rate_limit(rpm), "rpm")
        self.assertTrue(orch.is_rpm_rate_limit_error(rpm))
        self.assertFalse(orch.is_tpm_rate_limit_error(rpm))
        self.assertFalse(orch.is_tpd_rate_limit_error(rpm))
        self.assertFalse(orch.is_daily_quota_exhausted(rpm))

        self.assertEqual(orch.classify_groq_rate_limit(rpd), "rpd")
        self.assertTrue(orch.is_rpd_rate_limit_error(rpd))
        self.assertTrue(orch.is_daily_quota_exhausted(rpd))
        self.assertFalse(orch.is_tpm_rate_limit_error(rpd))
        self.assertFalse(orch.is_tpd_rate_limit_error(rpd))
        self.assertFalse(orch.is_rpm_rate_limit_error(rpd))

    def test_429_tpm_compacts_and_retries(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception(
                    "Error code: 429 - Rate limit reached for model "
                    "`openai/gpt-oss-20b` on tokens per minute (TPM): "
                    "Limit 8000, Used 3433, Requested 5174. "
                    "Please try again in 0.01s."
                )
            self.assertNotIn("tool_choice", kwargs)
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Answer after TPM backoff.",
                            tool_calls=None,
                        )
                    )
                ],
                usage=MagicMock(total_tokens=150),
            )

        with patch("app.chatbot.time.sleep") as sleep_mock, patch.object(
            chatbot, "client"
        ) as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-429").clear()
            answer = chatbot.chat_with_ai(
                "What is RAG?",
                user_id="orch-429",
                request_id="req-429",
            )

        sleep_mock.assert_called()
        self.assertEqual(calls["n"], 2)
        self.assertIn("tpm backoff", answer.lower())

    def test_429_tpd_stops_without_retry_or_finalization_llm(self):
        chatbot._get_conversation_store("orch-tpd").clear()
        calls = {"n": 0}

        def create_mixed(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                tc = MagicMock()
                tc.id = "s1"
                tc.function.name = "search_code"
                tc.function.arguments = '{"search_text":"execute_tool"}'
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ],
                    usage=MagicMock(total_tokens=100),
                )
            raise Exception(
                "Error code: 429 - tokens per day (TPD): Limit 200000, "
                "Used 199000, Requested 2000. try again in 20m0s."
            )

        with patch.object(
            chatbot, "client"
        ) as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ), patch.object(
            chatbot,
            "execute_tool",
            return_value={
                "success": True,
                "result": (
                    '{"success": true, "matches": [{"file_path": '
                    '"app/chatbot.py", "line": 779, '
                    '"text": "def execute_tool(...):"}]}'
                ),
            },
        ):
            mock_client.chat.completions.create.side_effect = create_mixed
            answer = chatbot.chat_with_ai(
                "Find execute_tool and inspect its complete implementation",
                user_id="orch-tpd",
                request_id="req-tpd",
            )

        # One tool-planning call + one failing call — no TPD retry / finalize LLM.
        self.assertEqual(calls["n"], 2)
        lowered = answer.lower()
        self.assertIn("daily", lowered)
        self.assertIn("partial", lowered)
        self.assertIn("verified", lowered)
        # Must remain an honest partial (forced pre-read evidence is fine;
        # never claim a finished model synthesis).
        self.assertTrue(
            "partial" in lowered
            or "not presented as fully verified" in lowered,
            msg=answer[:500],
        )

    def test_offline_answer_marks_partial_not_fully_verified(self):
        text = orch.build_offline_evidence_answer(
            "Explain execute_tool",
            evidence_snippets=[
                "read_file: def execute_tool(...): validates then runs"
            ],
            reason="tpd",
            function_complete=True,
            pending_continuation=False,
        )
        self.assertIn("daily", text.lower())
        self.assertIn("Verified from read_file source", text)
        self.assertIn("partial", text.lower())
        # With read evidence present, still marked partial (no final synthesis).
        self.assertTrue(
            "not presented as fully verified" in text.lower()
            or "still **partial**" in text.lower()
            or "still partial" in text.lower()
        )


class TestDuplicateAndBudget(unittest.TestCase):
    def test_duplicate_tool_fingerprint_blocked(self):
        args = {"search_text": "agent loop", "file_path": ""}
        fp = orch.tool_fingerprint("search_code", args)
        skip, reason = orch.should_skip_duplicate_tool(
            "search_code",
            args,
            fingerprints={fp},
            prior_reads=[],
            tool_counts={"search_code": 1},
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "DUPLICATE_TOOL_CALL")

    def test_overlapping_read_detected(self):
        a = {"file_path": "app/chatbot.py", "start_line": 10, "end_line": 40}
        b = {"file_path": "app/chatbot.py", "start_line": 30, "end_line": 60}
        self.assertTrue(orch.overlapping_read_range(a, b))
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            b,
            fingerprints=set(),
            prior_reads=[a],
            tool_counts={"read_file": 1},
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "OVERLAPPING_FILE_READ")

    def test_sequential_continuation_not_overlapping(self):
        first = {
            "file_path": "app/chatbot.py",
            "start_line": 750,
            "end_line": 819,
        }
        # Adjacent next unread line (confirmed production case).
        cont = {
            "file_path": "app/chatbot.py",
            "start_line": 820,
            "end_line": 890,
        }
        self.assertFalse(orch.overlapping_read_range(first, cont))
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            cont,
            fingerprints=set(),
            prior_reads=[first],
            tool_counts={"read_file": 1},
        )
        self.assertFalse(skip)
        self.assertIsNone(reason)

        # One-line boundary overlap normalizes to next unread line.
        boundary = {
            "file_path": "app/chatbot.py",
            "start_line": 819,
            "end_line": 890,
        }
        self.assertTrue(orch.is_boundary_only_touch(first, boundary))
        self.assertFalse(orch.overlapping_read_range(first, boundary))
        normalized = orch.normalize_sequential_read_args(boundary, [first])
        self.assertEqual(normalized["line_start"], 820)
        self.assertEqual(set(normalized.keys()), {"file_path", "line_start", "line_end"})
        self.assertNotIn("start_line", normalized)
        self.assertNotIn("end_line", normalized)
        skip2, _ = orch.should_skip_duplicate_tool(
            "read_file",
            normalized,
            fingerprints=set(),
            prior_reads=[first],
            tool_counts={"read_file": 1},
        )
        self.assertFalse(skip2)

    def test_actual_bounds_prevent_false_overlap_after_span_limit(self):
        """Requested 750–830 may only return 750–819; next 820–890 must pass."""
        requested = {
            "file_path": "app/chatbot.py",
            "line_start": 750,
            "line_end": 830,
        }
        payload = {
            "success": True,
            "file_path": "app/chatbot.py",
            "line_start": 750,
            "line_end": 819,
            "next_line": 820,
        }
        actual = orch.actual_read_bounds_from_result(requested, payload)
        self.assertEqual(actual["line_end"], 819)
        self.assertNotIn("start_line", actual)
        self.assertNotIn("end_line", actual)
        nxt = {
            "file_path": "app/chatbot.py",
            "start_line": 820,
            "end_line": 890,
        }
        # Bug: comparing against requested end=830 falsely overlaps 820–890.
        self.assertTrue(orch.overlapping_read_range(requested, nxt))
        # Fix: comparing against actual end=819 allows sequential continuation.
        self.assertFalse(orch.overlapping_read_range(actual, nxt))
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            nxt,
            fingerprints=set(),
            prior_reads=[actual],
            tool_counts={"read_file": 1},
        )
        self.assertFalse(skip)
        self.assertIsNone(reason)

    def test_agent_limit_returns_partial_result(self):
        text = orch.build_partial_evidence_answer(
            "find loop",
            evidence_snippets=["search_code: chatbot.py match"],
            reason="agent_limit",
        ).lower()
        self.assertIn("could not complete", text)
        self.assertIn("partial", text)
        self.assertIn("verified", text)


class TestChatRoutingBehavior(unittest.TestCase):
    def _run_chat(self, message: str, user_id: str, mock_create):
        with patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ), patch.object(
            chatbot, "execute_tool", wraps=chatbot.execute_tool
        ) as exec_tool:
            mock_client.chat.completions.create.side_effect = mock_create
            chatbot._get_conversation_store(user_id).clear()
            answer = chatbot.chat_with_ai(
                message, user_id=user_id, request_id=f"req-{user_id}"
            )
            return answer, mock_client, exec_tool

    def test_greeting_calls_llm_without_tools(self):
        def create(**kwargs):
            self.assertNotIn("tools", kwargs)
            self.assertNotIn("tool_choice", kwargs)
            msg = MagicMock(content="Hi! I'm doing well — how can I help?", tool_calls=None)
            return MagicMock(choices=[MagicMock(message=msg)])

        answer, client, exec_tool = self._run_chat(
            "Hi, how are you?", "orch-hi", create
        )
        self.assertIn("help", answer.lower())
        exec_tool.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_technical_question_no_tools(self):
        def create(**kwargs):
            # Must omit tool definitions entirely — not tool_choice="none".
            self.assertNotIn("tools", kwargs)
            self.assertNotIn("tool_choice", kwargs)
            sys_msgs = [
                m.get("content", "")
                for m in (kwargs.get("messages") or [])
                if isinstance(m, dict) and m.get("role") == "system"
            ]
            joined = "\n".join(sys_msgs).lower()
            self.assertNotIn("search_memory", joined)
            msg = MagicMock(
                content=(
                    "RAG (Retrieval-Augmented Generation) retrieves relevant "
                    "documents and feeds them to a model for grounded answers."
                ),
                tool_calls=None,
            )
            return MagicMock(choices=[MagicMock(message=msg)])

        answer, client, exec_tool = self._run_chat(
            "What is RAG?", "orch-rag", create
        )
        self.assertIn("retrieval", answer.lower())
        exec_tool.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_codebase_investigation_uses_search_then_read(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            tools = kwargs.get("tools") or []
            names = {(t.get("function") or {}).get("name") for t in tools}
            self.assertTrue({"search_code", "read_file"}.issubset(names) or names <= {"search_code", "read_file"})
            if calls["n"] == 1:
                tc = MagicMock()
                tc.id = "c1"
                tc.function.name = "search_code"
                tc.function.arguments = '{"search_text":"agent loop"}'
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(content=None, tool_calls=[tc]))]
                )
            if calls["n"] == 2:
                tc = MagicMock()
                tc.id = "c2"
                tc.function.name = "read_file"
                tc.function.arguments = (
                    '{"file_path":"app/chatbot.py","start_line":1,"end_line":40}'
                )
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(content=None, tool_calls=[tc]))]
                )
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="The agent loop is in chatbot.py.",
                            tool_calls=None,
                        )
                    )
                ]
            )

        with patch.object(
            chatbot,
            "execute_tool",
            side_effect=[
                {"success": True, "result": "chatbot.py: agent loop"},
                {"success": True, "result": "for step in range(MAX_AGENT_STEPS):"},
            ],
        ) as exec_tool, patch.object(
            chatbot, "client"
        ) as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-code").clear()
            answer = chatbot.chat_with_ai(
                "Find my agent loop implementation",
                user_id="orch-code",
                request_id="req-code",
            )

        self.assertIn("agent loop", answer.lower())
        self.assertEqual(exec_tool.call_count, 2)
        self.assertEqual(exec_tool.call_args_list[0].args[0], "search_code")
        self.assertEqual(exec_tool.call_args_list[1].args[0], "read_file")

    def test_repeated_tool_request_not_reexecuted(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                tc1 = MagicMock()
                tc1.id = "a"
                tc1.function.name = "search_code"
                tc1.function.arguments = '{"search_text":"MAX_AGENT_STEPS"}'
                tc2 = MagicMock()
                tc2.id = "b"
                tc2.function.name = "search_code"
                tc2.function.arguments = '{"search_text":"MAX_AGENT_STEPS"}'
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc1, tc2])
                        )
                    ]
                )
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Done with one search.", tool_calls=None
                        )
                    )
                ]
            )

        with patch.object(
            chatbot, "execute_tool", return_value={"success": True, "result": "hit"}
        ) as exec_tool, patch.object(
            chatbot, "client"
        ) as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-dup").clear()
            chatbot.chat_with_ai(
                "Find MAX_AGENT_STEPS in the code",
                user_id="orch-dup",
                request_id="req-dup",
            )

        # Second identical call skipped → execute_tool once.
        self.assertEqual(exec_tool.call_count, 1)

    def test_multi_tool_request_keeps_tools_available(self):
        captured = {}

        def create(**kwargs):
            captured["tools"] = kwargs.get("tools")
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Used multiple tools path.", tool_calls=None
                        )
                    )
                ]
            )

        answer, _, _ = self._run_chat(
            "Search the code for calculator and also calculate 2+2",
            "orch-multi",
            create,
        )
        self.assertTrue(captured.get("tools"))
        names = {
            (t.get("function") or {}).get("name") for t in captured["tools"]
        }
        self.assertIn("calculator", names)
        self.assertIn("search_code", names)
        self.assertIn("Used multiple tools", answer)

    def test_general_question_skips_ltm_retrieval(self):
        load_mock = MagicMock(
            return_value={"memories": [{"content": "irrelevant"}], "conflicts": [], "summaries": []}
        )

        def create(**kwargs):
            self.assertTrue(kwargs.get("tools") in (None, []))
            self.assertNotIn("tool_choice", kwargs)
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Photosynthesis converts light to energy.",
                            tool_calls=None,
                        )
                    )
                ]
            )

        with patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory", load_mock
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-ltm-skip").clear()
            answer = chatbot.chat_with_ai(
                "What is photosynthesis?",
                user_id="orch-ltm-skip",
                request_id="req-ltm-skip",
            )

        load_mock.assert_not_called()
        self.assertIn("photosynthesis", answer.lower())

    def test_substantial_overlap_blocked_not_continuation(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                tc1 = MagicMock()
                tc1.id = "r1"
                tc1.function.name = "read_file"
                tc1.function.arguments = (
                    '{"file_path":"app/chatbot.py","start_line":800,"end_line":850}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc1])
                        )
                    ]
                )
            if calls["n"] == 2:
                # Interior overlap — must be blocked, not re-executed.
                tc2 = MagicMock()
                tc2.id = "r2"
                tc2.function.name = "read_file"
                tc2.function.arguments = (
                    '{"file_path":"app/chatbot.py","start_line":820,"end_line":880}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc2])
                        )
                    ]
                )
            if calls["n"] == 3:
                tc3 = MagicMock()
                tc3.id = "r3"
                tc3.function.name = "read_file"
                tc3.function.arguments = (
                    '{"file_path":"app/chatbot.py","start_line":825,"end_line":870}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc3])
                        )
                    ]
                )
            # Answer (finalization or normal) — never tool_choice=none with tools.
            if kwargs.get("tools"):
                self.assertNotEqual(kwargs.get("tool_choice"), "none")
            else:
                self.assertNotIn("tool_choice", kwargs)
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=(
                                "Agent loop lives around those lines in "
                                "chatbot.py based on the read evidence."
                            ),
                            tool_calls=None,
                        )
                    )
                ]
            )

        with patch.object(
            chatbot,
            "execute_tool",
            return_value={
                "success": True,
                "result": (
                    '{"success": true, "file_path": "app/chatbot.py", '
                    '"line_start": 800, "line_end": 850, '
                    '"needs_continuation": false}\ncontent:\n'
                    "for step in range(MAX_AGENT_STEPS):"
                ),
            },
        ) as exec_tool, patch.object(
            chatbot, "client"
        ) as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-overlap").clear()
            answer = chatbot.chat_with_ai(
                "Read the agent loop section already known in chatbot.py",
                user_id="orch-overlap",
                request_id="req-overlap",
            )

        self.assertEqual(exec_tool.call_count, 1)
        self.assertIn("agent loop", answer.lower())

    def test_sequential_reads_750_819_then_820_890_after_span_limit(self):
        """Complete-function request must continue from next_line=820."""
        calls = {"n": 0}
        executed_ranges = []

        def create(**kwargs):
            calls["n"] += 1
            # Tool-free = finalization (continuation is forced without LLM).
            if "tools" not in kwargs:
                self.assertNotIn("tool_choice", kwargs)
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(
                                content=(
                                    "Verified: execute_tool looks up the tool, "
                                    "validates arguments, runs it, and returns "
                                    "structured success or error payloads. "
                                    "Exception paths catch ValidationError and "
                                    "execution failures without crashing the loop."
                                ),
                                tool_calls=None,
                            )
                        )
                    ]
                )
            if calls["n"] == 1:
                tc = MagicMock()
                tc.id = "s1"
                tc.function.name = "search_code"
                tc.function.arguments = '{"search_text":"execute_tool"}'
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            if calls["n"] == 2:
                tc = MagicMock()
                tc.id = "r1"
                tc.function.name = "read_file"
                tc.function.arguments = (
                    '{"file_path":"app/chatbot.py",'
                    '"line_start":750,"line_end":830}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            # Extra planning rounds: premature stop — continuation already
            # forced pre-LLM when pending; otherwise nudge another read.
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=(
                                "Sure thing—just let me know what "
                                "you'd like to dig into!"
                            ),
                            tool_calls=None,
                        )
                    )
                ]
            )

        def fake_execute(name, args, user_id=None, request_id=None, **_kwargs):
            if name == "read_file":
                start = int(args.get("start_line") or args.get("line_start"))
                end = int(args.get("end_line") or args.get("line_end"))
                actual_end = min(end, start + 69)
                executed_ranges.append((start, actual_end))
                # Keep the first window incomplete so orchestration must
                # continue from next_line (execute_tool spans well past one
                # 70-line read).
                needs = len(executed_ranges) == 1
                meta = {
                    "success": True,
                    "file_path": "app/chatbot.py",
                    "line_start": start,
                    "line_end": actual_end,
                    "next_line": actual_end + 1,
                    "truncated": False,
                    "span_limited": needs,
                    "needs_continuation": needs,
                }
                if needs:
                    meta["continuation"] = {
                        "file_path": "app/chatbot.py",
                        "next_line": actual_end + 1,
                        "suggested_end_line": actual_end + 70,
                    }
                body = (
                    "def execute_tool(tool_name, tool_arguments):\n"
                    "    try:\n"
                    "        validated = model(**tool_arguments)\n"
                    "        return {'success': True}\n"
                    if not needs
                    else (
                        "def execute_tool(tool_name, tool_arguments):\n"
                        "    print(tool_name)\n"
                        "    try:\n"
                        "        validated = model(**tool_arguments)\n"
                    )
                )
                return {
                    "success": True,
                    "result": (
                        __import__("json").dumps(meta)
                        + "\ncontent:\n"
                        + body
                    ),
                }
            return {
                "success": True,
                "result": "app/chatbot.py:771: def execute_tool",
            }

        with patch.object(
            chatbot, "execute_tool", side_effect=fake_execute
        ) as exec_tool, patch.object(
            chatbot, "client"
        ) as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-seq").clear()
            answer = chatbot.chat_with_ai(
                "Find the execute_tool function and inspect its complete "
                "implementation. Explain tool lookup, argument validation, "
                "successful execution, and exception handling. Read "
                "sequential file sections when necessary.",
                user_id="orch-seq",
                request_id="req-seq",
            )

        # search + first read + forced continuation read
        self.assertGreaterEqual(exec_tool.call_count, 3)
        self.assertGreaterEqual(len(executed_ranges), 2)
        # Canonicalize may snap to the live execute_tool def; require that
        # the first window covers it and the second continues after it.
        self.assertLessEqual(executed_ranges[0][0], 806)
        self.assertGreaterEqual(executed_ranges[0][1], 806)
        self.assertEqual(executed_ranges[1][0], executed_ranges[0][1] + 1)
        self.assertIn("validation", answer.lower())
        self.assertIn("exception", answer.lower())
        self.assertNotIn("just let me know", answer.lower())

    def test_beginner_question_complete_concise_no_tools(self):
        def create(**kwargs):
            self.assertNotIn("tools", kwargs)
            self.assertNotIn("tool_choice", kwargs)
            est = orch.estimate_messages_tokens(kwargs.get("messages") or [])
            self.assertLess(est + orch.estimate_tools_tokens(None), 7800)
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=(
                                "A chatbot mostly talks. An agent can also "
                                "use tools and remember goals — like a "
                                "friend who answers vs an assistant who "
                                "books the party for you."
                            ),
                            tool_calls=None,
                        )
                    )
                ],
                usage=MagicMock(total_tokens=400),
            )

        answer, client, exec_tool = self._run_chat(
            "Explain the difference between an AI chatbot and an AI agent "
            "using a simple real-world example. Keep it beginner-friendly.",
            "orch-beginner",
            create,
        )
        exec_tool.assert_not_called()
        self.assertIn("chatbot", answer.lower())
        self.assertIn("agent", answer.lower())
        self.assertNotIn("```", answer)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_finalization_omits_tools_and_tool_choice(self):
        """force_final uses a clean transcript; never sends tool_choice."""
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                # Build enough evidence then let force_final kick in.
                name = "search_code" if calls["n"] == 1 else "read_file"
                tc = MagicMock()
                tc.id = f"c{calls['n']}"
                tc.function.name = name
                if name == "search_code":
                    tc.function.arguments = '{"search_text":"loop"}'
                else:
                    tc.function.arguments = (
                        '{"file_path":"app/chatbot.py",'
                        '"start_line":1,"end_line":40}'
                    )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            if calls["n"] == 3:
                tc = MagicMock()
                tc.id = "c3"
                tc.function.name = "read_file"
                tc.function.arguments = (
                    '{"file_path":"app/chatbot.py",'
                    '"start_line":41,"end_line":90}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            # Finalization call
            self.assertIsNone(kwargs.get("tools"))
            self.assertNotIn("tool_choice", kwargs)
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Complete answer from evidence.",
                            tool_calls=None,
                        )
                    )
                ]
            )

        with patch.object(
            chatbot,
            "execute_tool",
            return_value={"success": True, "result": "evidence chunk"},
        ), patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-fin").clear()
            answer = chatbot.chat_with_ai(
                "Find my agent loop implementation",
                user_id="orch-fin",
                request_id="req-fin",
            )

        self.assertIn("complete answer", answer.lower())
        # Last create call must not include tools/tool_choice.
        last_kwargs = mock_client.chat.completions.create.call_args_list[-1].kwargs
        self.assertIsNone(last_kwargs.get("tools"))
        self.assertNotIn("tool_choice", last_kwargs)

    def test_finalization_failure_returns_partial_no_retry(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                tc = MagicMock()
                tc.id = "s1"
                tc.function.name = "search_code"
                tc.function.arguments = '{"search_text":"x"}'
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            if calls["n"] == 2:
                tc = MagicMock()
                tc.id = "r1"
                tc.function.name = "read_file"
                tc.function.arguments = (
                    '{"file_path":"app/chatbot.py","start_line":1,"end_line":20}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            if calls["n"] == 3:
                tc = MagicMock()
                tc.id = "r2"
                tc.function.name = "read_file"
                tc.function.arguments = (
                    '{"file_path":"app/chatbot.py","start_line":21,"end_line":50}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            # Finalization fails once — must not be retried.
            raise Exception(
                "Error code: 400 - Tool choice is none, but model called a tool"
            )

        with patch.object(
            chatbot,
            "execute_tool",
            return_value={"success": True, "result": "search_code: hit"},
        ), patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-fin-fail").clear()
            answer = chatbot.chat_with_ai(
                "Find my agent loop implementation",
                user_id="orch-fin-fail",
                request_id="req-fin-fail",
            )

        # One finalization attempt after 3 tool steps — no second finalization retry.
        self.assertEqual(calls["n"], 4)
        self.assertIn("search_code: hit", answer.lower())

    def test_tool_choice_none_never_sent_with_tools(self):
        choice, nudge = chatbot.resolve_tool_choice(
            user_message="Find the agent loop",
            step=7,
            max_steps=8,
            calculator_used=False,
            force_final=True,
        )
        self.assertEqual(choice, "none")
        self.assertIsNotNone(nudge)

        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            tools = kwargs.get("tools")
            if tools:
                self.assertNotEqual(kwargs.get("tool_choice"), "none")
            else:
                self.assertNotIn("tool_choice", kwargs)
            if calls["n"] == 1:
                tc = MagicMock()
                tc.id = "s1"
                tc.function.name = "search_code"
                tc.function.arguments = '{"search_text":"agent loop"}'
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            if calls["n"] == 2:
                tc = MagicMock()
                tc.id = "r1"
                tc.function.name = "read_file"
                tc.function.arguments = (
                    '{"file_path":"app/chatbot.py","start_line":1,"end_line":40}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Loop found in chatbot.py.",
                            tool_calls=None,
                        )
                    )
                ]
            )

        with patch.object(
            chatbot,
            "execute_tool",
            side_effect=[
                {"success": True, "result": "chatbot.py:120 match"},
                {"success": True, "result": "for step in range(...):"},
            ],
        ), patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-tc").clear()
            answer = chatbot.chat_with_ai(
                "Find my agent loop implementation",
                user_id="orch-tc",
                request_id="req-tc",
            )

        self.assertIn("loop", answer.lower())
        self.assertGreaterEqual(calls["n"], 3)

    def test_tool_choice_400_recovers_without_crash(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception(
                    "Error code: 400 - Tool choice is none, but model called a tool. "
                    "Failed generation: tool_use_failed"
                )
            self.assertTrue(kwargs.get("tools") in (None, []))
            self.assertNotIn("tool_choice", kwargs)
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Recovered final answer from evidence.",
                            tool_calls=None,
                        )
                    )
                ]
            )

        answer, _, _ = self._run_chat(
            "Hi, how are you?",
            "orch-400",
            create,
        )
        self.assertIn("recovered", answer.lower())
        self.assertEqual(calls["n"], 2)

    def test_unexpected_tool_call_while_disabled_recovers(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            self.assertTrue(kwargs.get("tools") in (None, []))
            if calls["n"] == 1:
                tc = MagicMock()
                tc.id = "bad"
                tc.function.name = "search_code"
                tc.function.arguments = '{"search_text":"x"}'
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Safe recovery answer.",
                            tool_calls=None,
                        )
                    )
                ]
            )

        with patch.object(
            chatbot, "execute_tool"
        ) as exec_tool, patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-unexpected").clear()
            answer = chatbot.chat_with_ai(
                "What is RAG?",
                user_id="orch-unexpected",
                request_id="req-unexpected",
            )

        exec_tool.assert_not_called()
        self.assertIn("safe recovery", answer.lower())

    def test_agent_limit_returns_useful_partial(self):
        self.assertIn(
            "search_code: hit in chatbot.py",
            orch.build_partial_evidence_answer(
                "find loop",
                evidence_snippets=["search_code: hit in chatbot.py"],
                reason="agent_limit",
            ),
        )


class TestBlockedToolHelpers(unittest.TestCase):
    def test_blocked_result_is_success_not_failure(self):
        result = orch.build_blocked_tool_result(
            reason="OVERLAPPING_FILE_READ",
            prior_result="prior code snippet",
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["blocked"])
        payload = __import__("json").loads(result["result"])
        self.assertTrue(payload["success"])
        self.assertTrue(payload["blocked"])
        self.assertIn("prior_result_preview", payload)

    def test_force_final_after_search_and_multiple_reads(self):
        # Single search+read is not enough — allow continuation reads.
        self.assertFalse(
            orch.should_force_final_answer(
                step=2,
                max_steps=8,
                successful_searches=1,
                successful_reads=1,
                total_successful_tools=2,
                blocked_in_a_row=0,
            )
        )
        self.assertTrue(
            orch.should_force_final_answer(
                step=3,
                max_steps=8,
                successful_searches=1,
                successful_reads=2,
                total_successful_tools=3,
                blocked_in_a_row=0,
                requirements_met=True,
            )
        )
        # Pending continuation blocks early finalization.
        self.assertFalse(
            orch.should_force_final_answer(
                step=3,
                max_steps=8,
                successful_searches=1,
                successful_reads=1,
                total_successful_tools=2,
                blocked_in_a_row=0,
                pending_continuation=True,
                requirements_met=False,
            )
        )
        # Overlapping-read streak must not finalize incomplete investigations.
        self.assertFalse(
            orch.should_force_final_answer(
                step=5,
                max_steps=8,
                successful_searches=2,
                successful_reads=1,
                total_successful_tools=3,
                blocked_in_a_row=2,
                requirements_met=False,
            )
        )

    def test_parse_continuation_and_validate_answer(self):
        result = {
            "success": True,
            "result": (
                '{"success": true, "file_path": "app/chatbot.py", '
                '"line_start": 750, "line_end": 819, "next_line": 820, '
                '"needs_continuation": true, "span_limited": true, '
                '"continuation": {"file_path": "app/chatbot.py", '
                '"next_line": 820, "suggested_end_line": 889}}'
                "\ncontent:\n    return serialized\n"
            ),
        }
        cont = orch.parse_read_file_continuation(result)
        self.assertIsNotNone(cont)
        self.assertEqual(cont["next_line"], 820)
        self.assertEqual(cont["suggested_end_line"], 889)

        req = orch.derive_investigation_requirements(
            "Inspect the complete implementation and explain exception handling"
        )
        self.assertTrue(req["needs_complete_function"])
        self.assertIn("exception_path", req["topics"])

        self.assertTrue(
            orch.is_premature_or_generic_answer(
                "Sure thing—just let me know what you'd like!",
                requirements=req,
                evidence_snippets=["read_file: def execute_tool"],
            )
        )
        cleaned = orch.validate_final_answer_text(
            "Intro\n```python\ndef foo():\n    x = 1\n"
        )
        self.assertNotIn("```", cleaned)
        self.assertIn("incomplete code example was omitted", cleaned.lower())

    def test_final_answer_validation_does_not_clip_complete_long_prose(self):
        answer = ("Complete answer paragraph. " * 180).strip()
        self.assertGreater(len(answer), 2800)
        self.assertEqual(orch.validate_final_answer_text(answer), answer)

    def test_finalization_messages_require_grounding(self):
        msgs = orch.build_finalization_messages(
            user_message="Explain execute_tool",
            evidence_snippets=[
                "read_file: lines 750-819 show def execute_tool(...)"
            ],
        )
        blob = " ".join(
            str(m.get("content") or "") for m in msgs if isinstance(m, dict)
        ).lower()
        self.assertIn("verified", blob)
        self.assertIn("inferred", blob)
        self.assertIn("do not invent", blob)
        self.assertIn("source_excerpts", blob)
        self.assertIn("tools are disabled", blob)
        # Must not ship the full tool-catalog system prompt.
        self.assertNotIn("search_memory", blob)

    def test_intent_needs_ltm_only_for_memory(self):
        self.assertFalse(
            orch.intent_needs_long_term_memory(
                orch.INTENT_TECHNICAL, "What is RAG?"
            )
        )
        self.assertFalse(
            orch.intent_needs_long_term_memory(
                orch.INTENT_CODEBASE, "Find my agent loop"
            )
        )
        self.assertTrue(
            orch.intent_needs_long_term_memory(
                orch.INTENT_MEMORY, "What do you remember about me?"
            )
        )
        self.assertTrue(
            orch.intent_needs_long_term_memory(
                orch.INTENT_TECHNICAL, "What is my project?"
            )
        )
        self.assertTrue(
            orch.intent_needs_long_term_memory(
                orch.INTENT_TECHNICAL,
                "What programming language did I ask you to remember?",
            )
        )
        self.assertTrue(
            orch.intent_needs_long_term_memory(
                orch.INTENT_GENERAL,
                "Did I tell you my preferred database is MongoDB?",
            )
        )
        self.assertTrue(
            orch.intent_needs_long_term_memory(
                orch.INTENT_TECHNICAL,
                "Which programming language do I prefer?",
            )
        )
        self.assertTrue(
            orch.intent_needs_long_term_memory(
                orch.INTENT_TECHNICAL,
                "What project am I working on?",
            )
        )
        self.assertFalse(
            orch.intent_needs_long_term_memory(
                orch.INTENT_TECHNICAL,
                "How do I choose a programming language for a project?",
            )
        )

    def test_is_tool_choice_error(self):
        self.assertTrue(
            orch.is_tool_choice_error(
                Exception("400 Tool choice is none, but model called a tool")
            )
        )
        self.assertFalse(
            orch.is_tool_choice_error(Exception("413 request too large"))
        )


class TestInvestigationCompleteness(unittest.TestCase):
    def test_educational_question_zero_tools(self):
        msg = (
            "Explain how an AI agent can identify, investigate, and fix "
            "a software bug with a real-world example"
        )
        intent = orch.classify_intent(msg)
        self.assertEqual(intent, orch.INTENT_TECHNICAL)
        self.assertFalse(orch.intent_allows_tools(intent))
        self.assertEqual(
            orch.tools_for_intent(intent, chatbot.tool_definitions),
            [],
        )

    def test_rate_limit_investigation_requires_all_four_categories(self):
        req = orch.derive_investigation_requirements(
            "Inspect my backend's intent router and Groq error-handling. "
            "Explain educational vs debugging and how TPM, TPD, RPM, and "
            "RPD errors are detected and handled. Read the relevant source."
        )
        ids = {t["id"] for t in req["required_targets"]}
        self.assertIn("classify_intent", ids)
        self.assertIn("classify_groq_rate_limit", ids)
        self.assertIn("is_tpm_rate_limit_error", ids)
        self.assertIn("is_tpd_rate_limit_error", ids)
        self.assertIn("is_rpm_rate_limit_error", ids)
        self.assertIn("is_rpd_rate_limit_error", ids)
        self.assertIn("rate_limit_retry_handling", ids)
        self.assertEqual(
            set(req["rate_limit_categories"]),
            {"TPM", "TPD", "RPM", "RPD"},
        )
        self.assertTrue(req["needs_source_reads"])
        self.assertGreaterEqual(req["min_source_reads"], 3)

    def test_search_discovery_is_not_verification(self):
        req = orch.derive_investigation_requirements(
            "Read classify_groq_rate_limit and is_tpm_rate_limit_error source"
        )
        evidence = orch.new_investigation_evidence_state()
        matches = [
            {
                "file_path": "app/services/agent_orchestration.py",
                "line": 515,
                "text": "def classify_groq_rate_limit(exc):",
            },
            {
                "file_path": "app/services/agent_orchestration.py",
                "line": 535,
                "text": "def is_tpm_rate_limit_error(exc):",
            },
        ]
        suggestions = orch.register_search_discoveries(
            evidence, req, matches
        )
        self.assertGreaterEqual(len(suggestions), 1)
        self.assertIn("classify_groq_rate_limit", evidence["discovered"])
        # Discovery alone does not satisfy requirements.
        self.assertFalse(
            orch.investigation_requirements_met(
                req,
                successful_searches=1,
                successful_reads=0,
                pending_continuation=False,
                calculator_used=False,
                function_complete=False,
                evidence=evidence,
            )
        )
        missing = orch.uncovered_required_targets(req, evidence)
        self.assertIn("classify_groq_rate_limit", missing)

    def test_reading_multiple_implementations_marks_verified(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent and classify_groq_rate_limit implementations"
        )
        evidence = orch.new_investigation_evidence_state()
        import ast
        from pathlib import Path

        def _span(name: str) -> tuple[int, int]:
            path = Path("app/services/agent_orchestration.py")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if (
                    isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    )
                    and node.name == name
                ):
                    return int(node.lineno), int(
                        getattr(node, "end_lineno", None) or node.lineno
                    )
            raise AssertionError(name)

        intent_start, intent_end = _span("classify_intent")
        groq_start, groq_end = _span("classify_groq_rate_limit")
        path = "app/services/agent_orchestration.py"
        orch.register_search_discoveries(
            evidence,
            req,
            [
                {
                    "file_path": path,
                    "line": intent_start,
                    "text": "def classify_intent(user_message):",
                },
                {
                    "file_path": path,
                    "line": groq_start,
                    "text": "def classify_groq_rate_limit(exc):",
                },
            ],
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path=path,
            line_start=intent_start,
            line_end=intent_end,
            content=(
                "def classify_intent(user_message):\n"
                "    if educational:\n"
                "        return INTENT_TECHNICAL\n"
            ),
            function_complete=True,
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path=path,
            line_start=groq_start,
            line_end=groq_end,
            content=(
                "def classify_groq_rate_limit(exc):\n"
                "    if 'tpd' in text: return 'tpd'\n"
                "    if 'tpm' in text: return 'tpm'\n"
                "    if 'rpm' in text: return 'rpm'\n"
                "    if 'rpd' in text: return 'rpd'\n"
            ),
            function_complete=True,
        )
        self.assertIn("classify_intent", evidence["verified_targets"])
        self.assertIn(
            "classify_groq_rate_limit", evidence["verified_targets"]
        )
        self.assertTrue(
            orch.investigation_requirements_met(
                req,
                successful_searches=1,
                successful_reads=2,
                pending_continuation=False,
                calculator_used=False,
                function_complete=True,
                evidence=evidence,
            )
        )

    def test_overlapping_read_prevention_keeps_investigation_open(self):
        prior = {
            "file_path": "app/services/agent_orchestration.py",
            "line_start": 180,
            "line_end": 202,
        }
        overlapping = {
            "file_path": "app/services/agent_orchestration.py",
            "line_start": 150,
            "line_end": 220,
        }
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            overlapping,
            fingerprints=set(),
            prior_reads=[prior],
            tool_counts={"read_file": 1},
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "OVERLAPPING_FILE_READ")
        # Incomplete requirements must not force-final after overlap streak.
        self.assertFalse(
            orch.should_force_final_answer(
                step=4,
                max_steps=8,
                successful_searches=2,
                successful_reads=1,
                total_successful_tools=3,
                blocked_in_a_row=2,
                requirements_met=False,
            )
        )

    def test_offline_answer_distinguishes_verified_inferred_missing(self):
        req = orch.derive_investigation_requirements(
            "Explain classify_groq_rate_limit and TPM TPD RPM RPD handling. "
            "Read the relevant source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/agent_orchestration.py",
            line_start=515,
            line_end=540,
            content=(
                "def classify_groq_rate_limit(exc):\n"
                "    return 'tpm'\n"
            ),
        )
        # Fix types if string leaked — register_read_verification casts int.
        text = orch.build_offline_evidence_answer(
            "Explain rate limits",
            evidence_snippets=[
                "search_code: def is_tpd_rate_limit_error",
                "read_file: def classify_groq_rate_limit",
            ],
            reason="requirements_incomplete",
            requirements=req,
            evidence_state=evidence,
        ).lower()
        self.assertIn("verified from read_file", text)
        self.assertIn("discovered via search_code", text)
        self.assertIn("missing_source_evidence", text)
        # Unread helpers must be called out.
        self.assertTrue(
            "is_tpd_rate_limit_error" in text
            or "is_tpm_rate_limit_error" in text
        )
        self.assertNotIn("fully verified rewrite", text)

    def test_all_four_rate_limit_classifiers_exist(self):
        samples = {
            "tpm": (
                "Error code: 429 - tokens per minute (TPM): Limit 8000"
            ),
            "tpd": (
                "Error code: 429 - tokens per day (TPD): Limit 200000"
            ),
            "rpm": (
                "Error code: 429 - requests per minute (RPM): Limit 30"
            ),
            "rpd": (
                "Error code: 429 - requests per day (RPD): Limit 14400"
            ),
        }
        for expected, msg in samples.items():
            err = Exception(msg)
            self.assertEqual(orch.classify_groq_rate_limit(err), expected)
        self.assertTrue(
            orch.is_tpm_rate_limit_error(Exception(samples["tpm"]))
        )
        self.assertTrue(
            orch.is_tpd_rate_limit_error(Exception(samples["tpd"]))
        )
        self.assertTrue(
            orch.is_rpm_rate_limit_error(Exception(samples["rpm"]))
        )
        self.assertTrue(
            orch.is_rpd_rate_limit_error(Exception(samples["rpd"]))
        )
        self.assertTrue(
            orch.is_daily_quota_exhausted(Exception(samples["tpd"]))
        )
        self.assertTrue(
            orch.is_daily_quota_exhausted(Exception(samples["rpd"]))
        )
        self.assertFalse(
            orch.is_daily_quota_exhausted(Exception(samples["tpm"]))
        )


class TestReadBudgetAndMultiFile(unittest.TestCase):
    def _read_args(self, path, start, end):
        return {
            "file_path": path,
            "line_start": start,
            "line_end": end,
            "start_line": start,
            "end_line": end,
        }

    def test_three_sequential_reads_same_file_allowed(self):
        path = "app/services/agent_orchestration.py"
        prior = []
        counts = {"read_file": 0}
        by_file = {}
        windows = [(100, 169), (170, 239), (240, 309)]
        for start, end in windows:
            args = self._read_args(path, start, end)
            skip, reason = orch.should_skip_duplicate_tool(
                "read_file",
                args,
                fingerprints=set(),
                prior_reads=prior,
                tool_counts=counts,
                read_counts_by_file=by_file,
            )
            self.assertFalse(skip, msg=reason)
            prior.append(args)
            counts["read_file"] = counts.get("read_file", 0) + 1
            key = path.lower()
            by_file[key] = by_file.get(key, 0) + 1
        self.assertEqual(counts["read_file"], 3)
        self.assertEqual(by_file[path.lower()], 3)

    def test_second_file_readable_after_three_reads_of_first(self):
        first = "app/services/agent_orchestration.py"
        second = "app/chatbot.py"
        prior = [
            self._read_args(first, 100, 169),
            self._read_args(first, 170, 239),
            self._read_args(first, 240, 309),
        ]
        counts = {"read_file": 3}
        by_file = {first.lower(): 3}
        args = self._read_args(second, 760, 829)
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            args,
            fingerprints=set(),
            prior_reads=prior,
            tool_counts=counts,
            read_counts_by_file=by_file,
        )
        self.assertFalse(
            skip,
            msg=f"second file must not hit global 3-read cap; got {reason}",
        )

    def test_duplicate_and_overlap_still_blocked(self):
        path = "app/chatbot.py"
        prior = [self._read_args(path, 750, 819)]
        counts = {"read_file": 1}
        by_file = {path.lower(): 1}
        dup = self._read_args(path, 750, 819)
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            dup,
            fingerprints={orch.tool_fingerprint("read_file", dup)},
            prior_reads=prior,
            tool_counts=counts,
            read_counts_by_file=by_file,
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "DUPLICATE_TOOL_CALL")

        overlap = self._read_args(path, 780, 850)
        skip2, reason2 = orch.should_skip_duplicate_tool(
            "read_file",
            overlap,
            fingerprints=set(),
            prior_reads=prior,
            tool_counts=counts,
            read_counts_by_file=by_file,
        )
        self.assertTrue(skip2)
        self.assertEqual(reason2, "OVERLAPPING_FILE_READ")

    def test_per_file_cap_blocks_fifth_read_same_file(self):
        path = "app/chatbot.py"
        prior = []
        counts = {"read_file": 0}
        by_file = {path.lower(): orch.MAX_READS_PER_FILE}
        args = self._read_args(path, 900, 969)
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            args,
            fingerprints=set(),
            prior_reads=prior,
            tool_counts=counts,
            read_counts_by_file=by_file,
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "PER_FILE_READ_LIMIT")

    def test_multi_file_symbols_require_all_reads_before_met(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent and execute_tool implementations "
            "across the backend. Read the relevant source."
        )
        ids = {t["id"] for t in req["required_targets"]}
        self.assertIn("classify_intent", ids)
        self.assertIn("execute_tool", ids)
        evidence = orch.new_investigation_evidence_state()
        orch.register_search_discoveries(
            evidence,
            req,
            [
                {
                    "file_path": "app/services/agent_orchestration.py",
                    "line": 205,
                    "text": "def classify_intent(user_message):",
                },
                {
                    "file_path": "app/chatbot.py",
                    "line": 785,
                    "text": "def execute_tool(tool_name, tool_arguments):",
                },
            ],
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/agent_orchestration.py",
            line_start=205,
            line_end=260,
            content="def classify_intent(user_message):\n    return INTENT_TECHNICAL\n",
        )
        self.assertFalse(
            orch.investigation_requirements_met(
                req,
                successful_searches=1,
                successful_reads=1,
                pending_continuation=False,
                calculator_used=False,
                function_complete=True,
                evidence=evidence,
            )
        )
        self.assertIn(
            "execute_tool",
            orch.uncovered_required_targets(req, evidence),
        )
        nxt = orch.next_unread_suggested_read(req, evidence)
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt["target"], "execute_tool")
        self.assertIn("chatbot.py", nxt["file_path"].replace("\\", "/"))

        orch.register_read_verification(
            evidence,
            req,
            file_path="app/chatbot.py",
            line_start=785,
            line_end=850,
            content=(
                "def execute_tool(tool_name, tool_arguments):\n"
                "    return {'success': True}\n"
            ),
        )
        self.assertTrue(
            orch.investigation_requirements_met(
                req,
                successful_searches=1,
                successful_reads=2,
                pending_continuation=False,
                calculator_used=False,
                function_complete=True,
                evidence=evidence,
            )
        )

    def test_partial_answer_lists_missing_when_budget_hit(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent and execute_tool. Read the source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/agent_orchestration.py",
            line_start=205,
            line_end=240,
            content="def classify_intent(user_message):\n    return x\n",
        )
        text = orch.build_offline_evidence_answer(
            "Inspect both",
            evidence_snippets=["read_file: def classify_intent"],
            reason="agent_limit",
            requirements=req,
            evidence_state=evidence,
        ).lower()
        self.assertIn("missing_source_evidence", text)
        self.assertIn("execute_tool", text)
        self.assertIn("verified from read_file", text)


class TestReadFileCanonicalArgs(unittest.TestCase):
    def test_canonical_args_validate_with_pydantic(self):
        from app.validation.tool_models import ReadFileInput

        args = orch.canonicalize_read_file_args({
            "file_path": "app/services/agent_orchestration.py",
            "line_start": 520,
            "line_end": 590,
            "start_line": 520,  # legacy noise must be dropped
            "end_line": 590,
        })
        self.assertEqual(
            set(args.keys()),
            {"file_path", "line_start", "line_end"},
        )
        model = ReadFileInput(**args)
        self.assertEqual(model.line_start, 520)
        self.assertEqual(model.line_end, 590)

    def test_sequential_continuation_from_539_validates(self):
        from app.validation.tool_models import ReadFileInput

        prior = [{
            "file_path": "app/services/agent_orchestration.py",
            "line_start": 523,
            "line_end": 538,
        }]
        requested = {
            "file_path": "app/services/agent_orchestration.py",
            "line_start": 538,
            "line_end": 620,
        }
        normalized = orch.normalize_sequential_read_args(requested, prior)
        self.assertEqual(normalized["line_start"], 539)
        self.assertEqual(normalized["line_end"], 620)
        self.assertNotIn("start_line", normalized)
        self.assertNotIn("end_line", normalized)
        ReadFileInput(**normalized)  # must not raise

    def test_forced_read_args_validate(self):
        from app.validation.tool_models import ReadFileInput

        forced = orch.canonicalize_read_file_args({
            "file_path": "app/services/agent_orchestration.py",
            "line_start": 553,
            "line_end": 622,
            "start_line": 553,
            "end_line": 622,
            "target": "is_rpd_rate_limit_error",  # internal meta must drop
        })
        self.assertEqual(
            set(forced.keys()),
            {"file_path", "line_start", "line_end"},
        )
        ReadFileInput(**forced)

    def test_extra_fields_still_rejected_by_schema(self):
        from pydantic import ValidationError
        from app.validation.tool_models import ReadFileInput

        with self.assertRaises(ValidationError) as ctx:
            ReadFileInput(
                file_path="app/chatbot.py",
                line_start=1,
                line_end=40,
                start_line=1,
            )
        self.assertIn("extra_forbidden", str(ctx.exception))

    def test_rate_limit_helpers_seeded_without_budget_exhaustion(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_groq_rate_limit and TPM TPD RPM RPD helpers. "
            "Read the relevant source."
        )
        evidence = orch.new_investigation_evidence_state()
        seeded = orch.seed_known_symbol_suggestions(req, evidence)
        ids = {s["target"] for s in seeded}
        for needed in (
            "classify_groq_rate_limit",
            "is_tpm_rate_limit_error",
            "is_tpd_rate_limit_error",
            "is_rpm_rate_limit_error",
            "is_rpd_rate_limit_error",
        ):
            self.assertIn(needed, ids)
        # Each suggestion is a single non-overlapping canonical window —
        # well under MAX_READ_FILE_TOTAL.
        self.assertLessEqual(len(seeded), orch.MAX_READ_FILE_TOTAL)
        for sug in seeded:
            self.assertIn("line_start", sug)
            self.assertIn("line_end", sug)
            self.assertNotIn("start_line", sug)
            self.assertNotIn("end_line", sug)


class TestFinalizationApiBoundary(unittest.TestCase):
    def test_build_kwargs_omits_tools_and_tool_choice(self):
        kwargs = orch.build_chat_completion_kwargs(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=200,
            tools=[{"type": "function"}],
            tool_choice="none",
            allow_tools=False,
        )
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)
        orch.assert_no_tools_in_request(kwargs)

    def test_assert_no_tools_rejects_leaks(self):
        with self.assertRaises(AssertionError):
            orch.assert_no_tools_in_request(
                {"model": "x", "tools": [], "messages": []}
            )
        with self.assertRaises(AssertionError):
            orch.assert_no_tools_in_request(
                {"model": "x", "tool_choice": "none", "messages": []}
            )

    def test_finalization_messages_neutralize_tool_imperatives(self):
        msgs = orch.build_finalization_messages(
            user_message="Search and read classify_groq_rate_limit source",
            evidence_snippets=[
                "read_file: def classify_groq_rate_limit(exc): return 'tpm'"
            ],
            missing_targets=["is_rpd_rate_limit_error"],
        )
        blob = " ".join(
            str(m.get("content") or "") for m in msgs
        ).lower()
        self.assertIn("tools are disabled", blob)
        self.assertIn("do not execute", blob)
        self.assertIn("source_excerpts", blob)
        self.assertIn("missing_source_evidence", blob)
        self.assertIn("is_rpd_rate_limit_error", blob)

    def test_finalization_create_never_passes_tools_or_tool_choice(self):
        """Live regression: force_final Groq call must omit both params."""
        seen_final = {"kwargs": None}
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                name = "search_code" if calls["n"] == 1 else "read_file"
                tc = MagicMock()
                tc.id = f"c{calls['n']}"
                tc.function.name = name
                tc.function.arguments = (
                    '{"search_text":"classify_intent"}'
                    if name == "search_code"
                    else (
                        '{"file_path":"app/services/agent_orchestration.py",'
                        '"line_start":213,"line_end":280}'
                    )
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            if calls["n"] == 3:
                tc = MagicMock()
                tc.id = "c3"
                tc.function.name = "read_file"
                tc.function.arguments = (
                    '{"file_path":"app/services/agent_orchestration.py",'
                    '"line_start":281,"line_end":350}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            # Finalization — fail the test if tools leak.
            seen_final["kwargs"] = dict(kwargs)
            self.assertNotIn("tools", kwargs)
            self.assertNotIn("tool_choice", kwargs)
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Verified from source excerpts.",
                            tool_calls=None,
                        )
                    )
                ]
            )

        with patch.object(
            chatbot,
            "execute_tool",
            return_value={
                "success": True,
                "result": (
                    '{"success": true, "file_path": '
                    '"app/services/agent_orchestration.py", '
                    '"line_start": 213, "line_end": 280, '
                    '"function_complete": true}\ncontent:\n'
                    "def classify_intent(user_message):\n    return INTENT_TECHNICAL\n"
                ),
            },
        ), patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-fin-api").clear()
            answer = chatbot.chat_with_ai(
                "Find classify_intent implementation",
                user_id="orch-fin-api",
                request_id="req-fin-api",
            )

        self.assertIsNotNone(seen_final["kwargs"])
        self.assertNotIn("tools", seen_final["kwargs"])
        self.assertNotIn("tool_choice", seen_final["kwargs"])
        self.assertIn("verified", answer.lower())

    def test_tool_choice_400_during_finalization_no_retry(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if "tools" in kwargs:
                tc = MagicMock()
                tc.id = f"t{calls['n']}"
                tc.function.name = (
                    "search_code" if calls["n"] == 1 else "read_file"
                )
                tc.function.arguments = (
                    '{"search_text":"x"}'
                    if tc.function.name == "search_code"
                    else (
                        '{"file_path":"app/chatbot.py",'
                        '"line_start":1,"line_end":40}'
                    )
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            # Tool-free finalization fails once — must not be retried.
            raise Exception(
                "Error code: 400 - Tool choice is none, but model called a tool"
            )

        with patch.object(
            chatbot,
            "execute_tool",
            return_value={
                "success": True,
                "result": (
                    '{"success": true, "file_path": "app/chatbot.py", '
                    '"line_start": 1, "line_end": 20}\ncontent:\n'
                    "def execute_tool():\n    return 1\n"
                ),
            },
        ), patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create
            chatbot._get_conversation_store("orch-fin-400").clear()
            answer = chatbot.chat_with_ai(
                "Find execute_tool and inspect its implementation",
                user_id="orch-fin-400",
                request_id="req-fin-400",
            )

        # Exactly one tool-free finalization attempt (no second create with
        # the same failing tool-free payload).
        tool_free_calls = [
            c.kwargs
            for c in mock_client.chat.completions.create.call_args_list
            if "tools" not in c.kwargs
        ]
        self.assertEqual(len(tool_free_calls), 1)
        self.assertNotIn("tool_choice", tool_free_calls[0])
        self.assertIn("verified", answer.lower())
        self.assertIn("tool-free finalization", answer.lower())

    def test_seed_known_rate_limit_symbols(self):
        req = orch.derive_investigation_requirements(
            "Explain TPM TPD RPM RPD handling. Read classify_groq_rate_limit "
            "and the is_*_rate_limit_error helpers."
        )
        evidence = orch.new_investigation_evidence_state()
        added = orch.seed_known_symbol_suggestions(req, evidence)
        targets = {s["target"] for s in added}
        self.assertIn("classify_groq_rate_limit", targets)
        self.assertTrue(
            {"is_tpm_rate_limit_error", "is_tpd_rate_limit_error"} & targets
        )


class TestFunctionBoundaryClip(unittest.TestCase):
    def test_indented_window_switches_to_next_def(self):
        from app.tools.file_reader import clip_to_function_boundary

        lines = (
            ["        return True\n"] * 8
            + ["\n"]
            + [
                "def classify_intent(user_message: str) -> str:\n",
                "    if educational:\n",
                "        return INTENT_TECHNICAL\n",
                "\n",
                "def other():\n",
                "    pass\n",
            ]
        )
        clipped, start, end, complete, _still = clip_to_function_boundary(
            lines, start_line=190
        )
        body = "".join(clipped)
        self.assertIn("def classify_intent", body)
        self.assertNotIn("def other", body)
        self.assertTrue(complete)
        self.assertEqual(start, 199)

    def test_mid_continuation_stops_before_next_def(self):
        """Long indented continuation must not jump to the next function."""
        from app.tools.file_reader import clip_to_function_boundary

        # 20 indented lines of the open body, then the next top-level def.
        lines = (
            ["    x = 1\n"] * 20
            + ["\n"]
            + [
                "def tools_for_intent(intent, defs):\n",
                "    return defs\n",
            ]
        )
        clipped, start, end, complete, still = clip_to_function_boundary(
            lines, start_line=280
        )
        body = "".join(clipped)
        self.assertNotIn("def tools_for_intent", body)
        self.assertIn("x = 1", body)
        self.assertEqual(start, 280)
        # Still open → ask for next_line continuation, not a jump.
        self.assertTrue(still)
        self.assertFalse(complete)
        self.assertEqual(end, 299)  # stopped before blank+def at offset 20

    def test_read_metadata_matches_content_lines(self):
        from app.tools import file_reader as fr

        # Use a real small project file window.
        result = fr.read_file(
            "app/services/agent_orchestration.py",
            line_start=1,
            line_end=20,
        )
        self.assertIsInstance(result, str)
        meta_text, _, body = result.partition("\ncontent:\n")
        payload = __import__("json").loads(meta_text)
        self.assertTrue(payload["success"])
        self.assertIn(
            payload["boundary_kind"],
            {
                "function_complete",
                "span_limited_open",
                "function_open",
                "span_limited_closed",
                "window_closed",
            },
        )
        self.assertEqual(
            payload["lines_read"],
            payload["content_line_count"],
        )
        self.assertTrue(payload["metadata_matches_content"])
        self.assertEqual(
            payload["line_end"] - payload["line_start"] + 1,
            payload["lines_read"],
        )
        # Displayed body line count must agree with metadata.
        displayed = body.count("\n") if body.endswith("\n") else (
            body.count("\n") + (1 if body.strip() else 0)
        )
        self.assertEqual(displayed, payload["content_line_count"])


class TestInvestigationBudgetsAndDedupe(unittest.TestCase):
    def _rate_limit_req(self):
        return orch.derive_investigation_requirements(
            "Inspect classify_groq_rate_limit and TPM TPD RPM RPD helpers "
            "is_tpm_rate_limit_error is_tpd_rate_limit_error "
            "is_rpm_rate_limit_error is_rpd_rate_limit_error. "
            "Read the relevant source."
        )

    def test_per_file_limit_scales_for_multi_symbol_file(self):
        path = "app/services/agent_orchestration.py"
        req = self._rate_limit_req()
        evidence = orch.new_investigation_evidence_state()
        orch.seed_known_symbol_suggestions(req, evidence)
        limit = orch.reads_per_file_limit(
            file_path=path,
            requirements=req,
            evidence=evidence,
        )
        # Base floor is 4; many helpers share this file → higher soft cap.
        self.assertGreater(limit, orch.MAX_READS_PER_FILE)
        self.assertGreaterEqual(limit, 6)

        prior = []
        counts = {"read_file": 0}
        by_file = {path.lower(): orch.MAX_READS_PER_FILE}
        # Fifth window must NOT be blocked when many required symbols live here.
        args = {
            "file_path": path,
            "line_start": 560,
            "line_end": 629,
        }
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            args,
            fingerprints=set(),
            prior_reads=prior,
            tool_counts=counts,
            read_counts_by_file=by_file,
            requirements=req,
            evidence=evidence,
            read_budget_total=orch.investigation_read_budget(req),
        )
        self.assertFalse(
            skip,
            msg=f"multi-symbol file must allow >{orch.MAX_READS_PER_FILE} "
            f"reads; got {reason}",
        )

    def test_per_file_floor_still_blocks_unrelated_fifth_read(self):
        path = "app/chatbot.py"
        by_file = {path.lower(): orch.MAX_READS_PER_FILE}
        skip, reason = orch.should_skip_duplicate_tool(
            "read_file",
            {"file_path": path, "line_start": 900, "line_end": 969},
            fingerprints=set(),
            prior_reads=[],
            tool_counts={"read_file": 0},
            read_counts_by_file=by_file,
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "PER_FILE_READ_LIMIT")

    def test_read_budget_covers_all_required_symbols(self):
        req = self._rate_limit_req()
        n = len(req["required_targets"])
        self.assertGreaterEqual(n, 5)
        budget = orch.investigation_read_budget(req)
        self.assertGreaterEqual(budget, n)
        self.assertGreaterEqual(budget, orch.MAX_READ_FILE_TOTAL)

    def test_finalization_reserve_forces_last_iteration(self):
        max_steps = 6
        # With default reserve=1, last open index is max_steps - 1.
        self.assertFalse(
            orch.should_force_final_answer(
                step=max_steps - 2,
                max_steps=max_steps,
                successful_searches=0,
                successful_reads=0,
                total_successful_tools=0,
                blocked_in_a_row=0,
                requirements_met=False,
            )
        )
        self.assertTrue(
            orch.should_force_final_answer(
                step=max_steps - 1,
                max_steps=max_steps,
                successful_searches=0,
                successful_reads=0,
                total_successful_tools=0,
                blocked_in_a_row=0,
                requirements_met=False,
            )
        )

    def test_forced_reads_prefer_rate_limit_symbols(self):
        req = self._rate_limit_req()
        evidence = orch.new_investigation_evidence_state()
        orch.seed_known_symbol_suggestions(req, evidence)
        nxt = orch.next_unread_suggested_read(
            req,
            evidence,
            prefer_targets=orch.RATE_LIMIT_SYMBOL_PRIORITY,
        )
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt["target"], "classify_groq_rate_limit")
        # After verifying the first, next prefer-target should follow priority.
        orch.register_read_verification(
            evidence,
            req,
            file_path=nxt["file_path"],
            line_start=nxt["line_start"],
            line_end=nxt["line_end"],
            content="def classify_groq_rate_limit(exc):\n    return 'tpm'\n",
        )
        nxt2 = orch.next_unread_suggested_read(
            req,
            evidence,
            prefer_targets=orch.RATE_LIMIT_SYMBOL_PRIORITY,
        )
        self.assertIsNotNone(nxt2)
        self.assertEqual(nxt2["target"], "is_tpm_rate_limit_error")

    def test_exclude_targets_skips_blocked_symbol(self):
        req = self._rate_limit_req()
        evidence = orch.new_investigation_evidence_state()
        orch.seed_known_symbol_suggestions(req, evidence)
        nxt = orch.next_unread_suggested_read(
            req,
            evidence,
            prefer_targets=orch.RATE_LIMIT_SYMBOL_PRIORITY,
            exclude_targets={"classify_groq_rate_limit"},
        )
        self.assertIsNotNone(nxt)
        self.assertNotEqual(nxt["target"], "classify_groq_rate_limit")
        self.assertEqual(nxt["target"], "is_tpm_rate_limit_error")

    def test_duplicate_search_of_discovered_symbol_skipped(self):
        req = self._rate_limit_req()
        evidence = orch.new_investigation_evidence_state()
        orch.register_search_discoveries(
            evidence,
            req,
            [
                {
                    "file_path": "app/services/agent_orchestration.py",
                    "line": 523,
                    "text": "def classify_groq_rate_limit(exc):",
                }
            ],
        )
        skip, reason = orch.should_skip_duplicate_tool(
            "search_code",
            {"search_text": "classify_groq_rate_limit"},
            fingerprints=set(),
            prior_reads=[],
            tool_counts={"search_code": 1},
            evidence=evidence,
            searched_queries=set(),
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "SEARCH_SKIPPED_KNOWN_SYMBOL")

    def test_searched_queries_dedupe_exact_repeat(self):
        searched = {orch.normalize_search_query("is_rpd_rate_limit_error")}
        skip, reason = orch.should_skip_duplicate_tool(
            "search_code",
            {"search_text": "is_rpd_rate_limit_error"},
            fingerprints=set(),
            prior_reads=[],
            tool_counts={"search_code": 1},
            searched_queries=searched,
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "SEARCH_SKIPPED_KNOWN_SYMBOL")

    def test_continuation_next_line_canonical_args(self):
        from app.validation.tool_models import ReadFileInput

        cont = {
            "file_path": "app/services/agent_orchestration.py",
            "next_line": 540,
            "suggested_end_line": 609,
        }
        args = orch.canonicalize_read_file_args({
            "file_path": cont["file_path"],
            "line_start": cont["next_line"],
            "line_end": cont["suggested_end_line"],
            "start_line": cont["next_line"],
            "end_line": cont["suggested_end_line"],
        })
        self.assertEqual(
            set(args.keys()),
            {"file_path", "line_start", "line_end"},
        )
        self.assertEqual(args["line_start"], 540)
        model = ReadFileInput(**args)
        self.assertEqual(model.line_start, 540)
        self.assertEqual(model.line_end, 609)

    def test_multiple_functions_same_file_all_readable(self):
        """Several required helpers in one module must each get a window."""
        path = "app/services/agent_orchestration.py"
        req = self._rate_limit_req()
        evidence = orch.new_investigation_evidence_state()
        orch.seed_known_symbol_suggestions(req, evidence)
        budget = orch.investigation_read_budget(req)
        file_limit = orch.reads_per_file_limit(
            file_path=path,
            requirements=req,
            evidence=evidence,
        )
        prior = []
        counts = {"read_file": 0}
        by_file: dict[str, int] = {}
        verified = 0
        while True:
            nxt = orch.next_unread_suggested_read(
                req,
                evidence,
                prefer_targets=orch.RATE_LIMIT_SYMBOL_PRIORITY,
            )
            if not nxt:
                break
            args = orch.canonicalize_read_file_args({
                "file_path": nxt["file_path"],
                "line_start": nxt["line_start"],
                "line_end": nxt["line_end"],
            })
            skip, reason = orch.should_skip_duplicate_tool(
                "read_file",
                args,
                fingerprints=set(),
                prior_reads=prior,
                tool_counts=counts,
                read_counts_by_file=by_file,
                requirements=req,
                evidence=evidence,
                read_budget_total=budget,
            )
            self.assertFalse(
                skip,
                msg=f"blocked {nxt['target']} after {verified} reads: {reason}",
            )
            orch.register_read_verification(
                evidence,
                req,
                file_path=args["file_path"],
                line_start=args["line_start"],
                line_end=args["line_end"],
                content=f"def {nxt['target']}(...):\n    return True\n",
            )
            prior.append(args)
            counts["read_file"] += 1
            key = path.lower()
            by_file[key] = by_file.get(key, 0) + 1
            verified += 1
            if verified > file_limit + 2:
                self.fail("runaway forced-read loop")
        # At least the core rate-limit cluster should be covered.
        for needed in (
            "classify_groq_rate_limit",
            "is_tpm_rate_limit_error",
            "is_tpd_rate_limit_error",
            "is_rpm_rate_limit_error",
            "is_rpd_rate_limit_error",
        ):
            self.assertIn(needed, evidence.get("verified_targets") or set())
        self.assertLessEqual(by_file.get(path.lower(), 0), file_limit)


if __name__ == "__main__":
    unittest.main()
