"""Topics 45–47 — tool timeouts, tool retries, and tool fallbacks."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "e2e-test-jwt-secret-key-32chars-min!!")

from app.services.request_cancellation import (
    CancellationToken,
    RequestCancelled,
)
from app.services.tool_execution import (
    PreparedToolCall,
    execute_tool_async,
    execute_tool_sync,
    execute_tools_bounded,
    reset_global_tool_pool_for_tests,
)
from app.services.tool_reliability import (
    ReliabilityClock,
    ToolFailureCategory,
    classify_tool_failure,
    invoke_async_once_with_timeout,
    invoke_sync_once_with_timeout,
    outstanding_timed_out_workers,
    reset_reliability_state_for_tests,
    run_tool_with_reliability,
    select_fallback_tool,
    should_retry_tool,
)


class Topic45TimeoutTests(unittest.TestCase):
    def setUp(self):
        reset_global_tool_pool_for_tests()
        reset_reliability_state_for_tests()

    def test_sync_success_before_timeout(self):
        def fast(name, args, **kwargs):
            return {"success": True, "result": "ok"}

        result = invoke_sync_once_with_timeout(
            "calculator",
            {},
            execute_fn=fast,
            timeout_seconds=2.0,
        )
        self.assertTrue(result["success"])

    def test_sync_caller_timeout_worker_may_still_run(self):
        started = threading.Event()
        finished = threading.Event()

        def slow(name, args, **kwargs):
            started.set()
            time.sleep(0.4)
            finished.set()
            return {"success": True, "result": "late"}

        result = invoke_sync_once_with_timeout(
            "calculator",
            {},
            execute_fn=slow,
            timeout_seconds=0.05,
            call_key="slow-1",
        )
        self.assertFalse(result["success"])
        self.assertEqual(result.get("error_code"), "TOOL_TIMEOUT")
        self.assertTrue(result.get("worker_may_still_run"))
        self.assertTrue(started.wait(1.0))
        # Slot remains accounted until worker finishes.
        self.assertGreaterEqual(outstanding_timed_out_workers(), 1)
        self.assertTrue(finished.wait(2.0))
        # Give callback a moment.
        time.sleep(0.05)
        self.assertEqual(outstanding_timed_out_workers(), 0)

    def test_async_timeout_cancels_coroutine(self):
        cancelled = {"n": 0}

        async def slow_exec(name, args, **kwargs):
            try:
                await asyncio.sleep(2.0)
                return {"success": True}
            except asyncio.CancelledError:
                cancelled["n"] += 1
                raise

        async def _run():
            return await invoke_async_once_with_timeout(
                "calculator",
                {},
                execute_fn=slow_exec,
                timeout_seconds=0.05,
            )

        result = asyncio.run(_run())
        self.assertFalse(result["success"])
        self.assertEqual(result.get("error_code"), "TOOL_TIMEOUT")
        self.assertFalse(result.get("worker_may_still_run"))
        self.assertGreaterEqual(cancelled["n"], 1)

    def test_async_success_before_timeout(self):
        async def quick(name, args, **kwargs):
            await asyncio.sleep(0.01)
            return {"success": True, "result": "async-ok"}

        async def _run():
            return await invoke_async_once_with_timeout(
                "calculator",
                {},
                execute_fn=quick,
                timeout_seconds=1.0,
            )

        result = asyncio.run(_run())
        self.assertTrue(result["success"])

    def test_timeout_no_false_success_in_parallel_batch(self):
        def flaky(name, args, **kwargs):
            if args.get("slow"):
                time.sleep(0.3)
            return {"success": True, "result": name}

        calls = [
            PreparedToolCall(0, "a", "calculator", {"slow": False}),
            PreparedToolCall(1, "b", "get_current_time", {"slow": True}),
        ]
        # Patch timeout for get_current_time via invoke path using short timeout
        # through reliability meta override by calling run directly.
        with patch(
            "app.services.tool_reliability.resolve_tool_timeout",
            side_effect=lambda n: 0.05 if n == "get_current_time" else 2.0,
        ):
            results = execute_tools_bounded(calls, execute_fn=flaky, max_concurrency=2)
        self.assertTrue(results[0]["success"])
        self.assertFalse(results[1]["success"])
        self.assertEqual(results[1].get("error_code"), "TOOL_TIMEOUT")


class Topic46RetryTests(unittest.TestCase):
    def setUp(self):
        reset_global_tool_pool_for_tests()
        reset_reliability_state_for_tests()

    def test_classify_categories(self):
        self.assertEqual(
            classify_tool_failure({"success": False, "error": "Invalid tool arguments"}),
            ToolFailureCategory.VALIDATION,
        )
        self.assertEqual(
            classify_tool_failure(
                {"success": False, "error": "Authenticated user_id is required"}
            ),
            ToolFailureCategory.AUTHORIZATION,
        )
        self.assertEqual(
            classify_tool_failure({"success": False, "error_code": "TOOL_TIMEOUT"}),
            ToolFailureCategory.TIMEOUT,
        )
        self.assertEqual(
            classify_tool_failure(None, exc=RequestCancelled("x")),
            ToolFailureCategory.CANCELLATION,
        )

    def test_transient_then_success(self):
        calls = {"n": 0}

        def flaky(name, args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"success": False, "error": "temporary network unavailable"}
            return {"success": True, "result": "ok"}

        clock = ReliabilityClock(sleep_fn=lambda s: None, rng=__import__("random").Random(0))
        result = run_tool_with_reliability(
            "calculator",
            {},
            execute_fn=flaky,
            clock=clock,
        )
        self.assertTrue(result["success"])
        self.assertEqual(calls["n"], 2)
        self.assertGreaterEqual(result["reliability"]["attempts"], 2)

    def test_validation_no_retry(self):
        calls = {"n": 0}

        def bad(name, args, **kwargs):
            calls["n"] += 1
            return {"success": False, "error": "Invalid tool arguments"}

        result = run_tool_with_reliability(
            "calculator",
            {},
            execute_fn=bad,
            clock=ReliabilityClock(sleep_fn=lambda s: None),
        )
        self.assertFalse(result["success"])
        self.assertEqual(calls["n"], 1)

    def test_auth_no_retry(self):
        calls = {"n": 0}

        def bad(name, args, **kwargs):
            calls["n"] += 1
            return {
                "success": False,
                "error": "Authenticated user_id is required for memory tools",
            }

        result = run_tool_with_reliability(
            "search_memory",
            {"query": "x"},
            execute_fn=bad,
            clock=ReliabilityClock(sleep_fn=lambda s: None),
        )
        self.assertEqual(calls["n"], 1)

    def test_cancel_during_backoff(self):
        token = CancellationToken(request_id="r1")
        calls = {"n": 0}

        def flaky(name, args, **kwargs):
            calls["n"] += 1
            return {"success": False, "error": "temporary unavailable"}

        def sleeper(seconds):
            token.cancel("client_disconnect")
            time.sleep(0.01)

        with self.assertRaises(RequestCancelled):
            run_tool_with_reliability(
                "calculator",
                {},
                execute_fn=flaky,
                cancellation=token,
                clock=ReliabilityClock(sleep_fn=sleeper),
            )
        self.assertEqual(calls["n"], 1)

    def test_retry_after_exceeds_budget(self):
        ok, wait, reason = should_retry_tool(
            tool_name="calculator",
            category=ToolFailureCategory.RATE_LIMIT,
            attempt=1,
            elapsed=1.0,
            retry_after=1000.0,
            worker_may_still_run=False,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "retry_after_exceeds_budget")

    def test_non_idempotent_not_retried(self):
        calls = {"n": 0}

        def fail(name, args, **kwargs):
            calls["n"] += 1
            return {"success": False, "error": "temporary unavailable"}

        result = run_tool_with_reliability(
            "save_memory",
            {"memory": "x"},
            execute_fn=fail,
            clock=ReliabilityClock(sleep_fn=lambda s: None),
        )
        self.assertFalse(result["success"])
        self.assertEqual(calls["n"], 1)

    def test_ambiguous_in_flight_not_retried(self):
        ok, _, reason = should_retry_tool(
            tool_name="save_memory",
            category=ToolFailureCategory.TIMEOUT,
            attempt=1,
            elapsed=0.0,
            retry_after=None,
            worker_may_still_run=True,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "worker_may_still_run")


class Topic47FallbackTests(unittest.TestCase):
    def setUp(self):
        reset_global_tool_pool_for_tests()
        reset_reliability_state_for_tests()

    def test_success_skips_fallback(self):
        names: list[str] = []

        def ok(name, args, **kwargs):
            names.append(name)
            return {"success": True, "result": name}

        result = run_tool_with_reliability(
            "search_memory",
            {"query": "q"},
            execute_fn=ok,
            permitted_tools={"search_memory", "search_memory_summaries"},
            registered_tools={"search_memory", "search_memory_summaries"},
        )
        self.assertTrue(result["success"])
        self.assertEqual(names, ["search_memory"])
        self.assertFalse(result["reliability"]["fallback_used"])

    def test_eligible_fallback_success(self):
        names: list[str] = []

        def route(name, args, **kwargs):
            names.append(name)
            if name == "search_memory":
                return {"success": False, "error": "temporary unavailable"}
            return {"success": True, "result": "summaries"}

        result = run_tool_with_reliability(
            "search_memory",
            {"query": "q"},
            execute_fn=route,
            permitted_tools={"search_memory", "search_memory_summaries"},
            registered_tools={"search_memory", "search_memory_summaries"},
            clock=ReliabilityClock(sleep_fn=lambda s: None),
        )
        self.assertTrue(result["success"])
        self.assertIn("search_memory_summaries", names)
        self.assertTrue(result["reliability"]["fallback_used"])
        self.assertEqual(
            result["reliability"]["executed_tool"], "search_memory_summaries"
        )
        self.assertEqual(result["reliability"]["original_tool"], "search_memory")
        self.assertFalse(result["reliability"].get("original_tool_succeeded", True))

    def test_unauthorized_fallback_rejected(self):
        fb, reason = select_fallback_tool(
            "search_memory",
            ToolFailureCategory.TRANSIENT,
            permitted_tools={"search_memory"},
            registered_tools={"search_memory", "search_memory_summaries"},
            visited=["search_memory"],
            depth=0,
        )
        self.assertIsNone(fb)
        self.assertEqual(reason, "fallback_not_permitted")

    def test_auth_failure_no_fallback(self):
        fb, reason = select_fallback_tool(
            "search_memory",
            ToolFailureCategory.AUTHORIZATION,
            permitted_tools={"search_memory", "search_memory_summaries"},
            registered_tools={"search_memory", "search_memory_summaries"},
            visited=["search_memory"],
            depth=0,
        )
        self.assertIsNone(fb)
        self.assertIn("authorization", reason or "")

    def test_cancel_prevents_fallback(self):
        token = CancellationToken(request_id="fb")
        calls = {"n": 0}

        def fail_then_cancel(name, args, **kwargs):
            calls["n"] += 1
            token.cancel("client_disconnect")
            return {"success": False, "error": "temporary unavailable"}

        with self.assertRaises(RequestCancelled):
            run_tool_with_reliability(
                "search_memory",
                {"query": "q"},
                execute_fn=fail_then_cancel,
                cancellation=token,
                permitted_tools={"search_memory", "search_memory_summaries"},
                registered_tools={"search_memory", "search_memory_summaries"},
                clock=ReliabilityClock(sleep_fn=lambda s: None),
            )

    def test_fallback_depth_limit(self):
        fb, reason = select_fallback_tool(
            "search_memory",
            ToolFailureCategory.TRANSIENT,
            permitted_tools={"search_memory", "search_memory_summaries"},
            registered_tools={"search_memory", "search_memory_summaries"},
            visited=["search_memory"],
            depth=99,
        )
        self.assertIsNone(fb)
        self.assertEqual(reason, "fallback_depth_exceeded")


class IntegrationReliabilityTests(unittest.TestCase):
    def setUp(self):
        reset_global_tool_pool_for_tests()
        reset_reliability_state_for_tests()

    def test_agent_loop_uses_reliability_on_multi_tool(self):
        from app import chatbot

        patches = [
            patch.object(chatbot, "client"),
            patch.object(
                chatbot,
                "load_long_term_memory",
                return_value={"memories": [], "conflicts": [], "summaries": []},
            ),
            patch.object(chatbot, "restore_task_context_from_redis"),
            patch.object(chatbot, "save_task_context_to_redis"),
            patch.object(chatbot, "try_save_explicit_memory", return_value=False),
            patch.object(chatbot, "try_update_explicit_memory", return_value=None),
            patch.object(chatbot, "try_delete_explicit_memory", return_value=None),
            patch.object(chatbot, "try_inspect_memory_metadata", return_value=None),
            patch(
                "app.services.memory_scope_service.get_active_project_scope",
                return_value={"project_id": None, "tenant_id": None},
            ),
            patch.object(chatbot, "add_to_conversation_memory"),
            patch.object(chatbot, "get_conversation_memory", return_value=[]),
        ]
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]
            c1 = MagicMock()
            c1.id = "t1"
            c1.function.name = "calculator"
            c1.function.arguments = json.dumps({"expression": "1+1"})
            c2 = MagicMock()
            c2.id = "t2"
            c2.function.name = "get_current_time"
            c2.function.arguments = json.dumps({})
            step1 = MagicMock()
            step1.tool_calls = [c1, c2]
            step1.content = None
            final = MagicMock()
            final.tool_calls = None
            final.content = "done"
            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            def ok(name, args, **kwargs):
                return {
                    "success": True,
                    "result": json.dumps({"success": True, "name": name}),
                }

            with patch.object(chatbot, "execute_tool", side_effect=ok), patch(
                "app.services.tool_reliability.run_tool_with_reliability",
                wraps=run_tool_with_reliability,
            ) as rel:
                chatbot.chat_with_ai(
                    "Calculate 1+1 and tell me the time.",
                    user_id="u_rel",
                    request_id="req_rel",
                )
            self.assertTrue(rel.called)
        finally:
            for p in patches:
                p.stop()

    def test_topic38_decide_llm_retry_untouched_by_tool_module(self):
        import inspect
        from app.services import agent_retry
        from app.services import tool_reliability

        src = inspect.getsource(tool_reliability)
        # Docstring may mention Topic 38; must not import or call it.
        self.assertNotIn("from app.services.agent_retry", src)
        self.assertNotIn("decide_llm_retry(", src)
        self.assertTrue(hasattr(agent_retry, "decide_llm_retry"))


if __name__ == "__main__":
    unittest.main()
