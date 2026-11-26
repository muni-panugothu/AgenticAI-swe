"""BD-E — Retry, fallback, timeout, and cancellation abuse."""

from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from app.services.request_cancellation import (  # noqa: E402
    CancellationToken,
    RequestCancelled,
)
from app.services.tool_execution import (  # noqa: E402
    PreparedToolCall,
    execute_tools_bounded,
    reset_global_tool_pool_for_tests,
)
from app.services.tool_reliability import (  # noqa: E402
    ReliabilityClock,
    ToolFailureCategory,
    classify_tool_failure,
    reset_reliability_state_for_tests,
    run_tool_with_reliability,
    select_fallback_tool,
    should_retry_tool,
)
from tests.security._fixtures import SecurityCase  # noqa: E402


class TestRetryFallbackCancel(SecurityCase):
    def setUp(self):
        super().setUp()
        reset_global_tool_pool_for_tests()
        reset_reliability_state_for_tests()

    def test_BD_E01_transient_retries_bounded(self):
        attempts = {"n": 0}

        def flaky(name, args, **kwargs):
            attempts["n"] += 1
            return {"success": False, "error": "temporary unavailable"}

        result = run_tool_with_reliability(
            "calculator",
            {"expression": "1"},
            execute_fn=flaky,
            clock=ReliabilityClock(sleep_fn=lambda s: None),
            permitted_tools={"calculator"},
            registered_tools={"calculator"},
        )
        self.assertFalse(result.get("success"))
        # max_retries=2 => attempt 1 + up to 2 retries => <= 3
        self.assertLessEqual(attempts["n"], 3)
        self.assertGreaterEqual(attempts["n"], 2)

    def test_BD_E02_permanent_error_no_retry(self):
        attempts = {"n": 0}

        def permanent(name, args, **kwargs):
            attempts["n"] += 1
            return {
                "success": False,
                "error": "Invalid tool arguments",
                "details": "schema validation failed",
            }

        result = run_tool_with_reliability(
            "calculator",
            {"expression": "1"},
            execute_fn=permanent,
            clock=ReliabilityClock(sleep_fn=lambda s: None),
        )
        self.assertEqual(attempts["n"], 1)
        self.assertEqual(
            result["reliability"]["failure_category"],
            ToolFailureCategory.VALIDATION.value,
        )

    def test_BD_E03_authorization_not_retried(self):
        ok, _, reason = should_retry_tool(
            tool_name="save_memory",
            category=ToolFailureCategory.AUTHORIZATION,
            attempt=1,
            elapsed=0.0,
            retry_after=None,
            worker_may_still_run=False,
        )
        self.assertFalse(ok)
        self.assertIn("authorization", reason)

    def test_BD_E04_approval_required_classified_as_authorization(self):
        cat = classify_tool_failure(
            {
                "success": False,
                "approval_required": True,
                "pending_human_review": True,
                "error_code": "APPROVAL_REQUIRED",
            }
        )
        self.assertEqual(cat, ToolFailureCategory.AUTHORIZATION)
        fb, reason = select_fallback_tool(
            "save_memory",
            cat,
            permitted_tools={"save_memory", "search_memory"},
            registered_tools={"save_memory", "search_memory"},
            visited=["save_memory"],
            depth=0,
        )
        self.assertIsNone(fb)
        self.assertIn("authorization", reason or "")

    def test_BD_E05_fallback_cycle_prevented(self):
        visited = ["search_memory", "search_memory_summaries"]
        fb, reason = select_fallback_tool(
            "search_memory_summaries",
            ToolFailureCategory.TRANSIENT,
            permitted_tools={"search_memory", "search_memory_summaries"},
            registered_tools={"search_memory", "search_memory_summaries"},
            visited=visited,
            depth=0,
        )
        self.assertIsNone(fb)
        self.assertIn(
            reason,
            {"no_eligible_fallback", "fallback_depth_exceeded"},
        )

    def test_BD_E06_cancel_during_retry_backoff(self):
        token = CancellationToken(request_id="bd-e06")
        attempts = {"n": 0}

        def flaky(name, args, **kwargs):
            attempts["n"] += 1
            token.cancel("client_disconnect")
            return {"success": False, "error": "temporary unavailable"}

        with self.assertRaises(RequestCancelled):
            run_tool_with_reliability(
                "calculator",
                {"expression": "1"},
                execute_fn=flaky,
                cancellation=token,
                clock=ReliabilityClock(sleep_fn=lambda s: None),
            )
        self.assertEqual(attempts["n"], 1)

    def test_BD_E07_cancel_stops_new_parallel_scheduling(self):
        token = CancellationToken(request_id="bd-e07")
        started = {"n": 0}
        barrier = threading.Barrier(2)

        def slow(name, args, **kwargs):
            started["n"] += 1
            if started["n"] == 1:
                barrier.wait(timeout=2)
                token.cancel("client_disconnect")
            return {"success": True, "result": name}

        calls = [
            PreparedToolCall(0, "c1", "calculator", {"expression": "1"}),
            PreparedToolCall(1, "c2", "calculator", {"expression": "2"}),
            PreparedToolCall(2, "c3", "get_current_time", {}),
        ]
        # First worker cancels; remaining must not all succeed as new work.
        try:
            results = execute_tools_bounded(
                calls,
                execute_fn=slow,
                cancellation=token,
                max_concurrency=1,
            )
        except RequestCancelled:
            results = []
        # Deterministic: with concurrency 1, at most the in-flight + maybe one more
        # before cancel check; never claim all three succeeded after cancel.
        if results:
            successes = [r for r in results if r.get("success")]
            self.assertLess(len(successes), 3)

    def test_BD_E08_sibling_success_preserved_on_peer_failure(self):
        def route(name, args, **kwargs):
            if name == "calculator":
                return {"success": False, "error": "boom", "error_code": "TOOL_ERROR"}
            return {"success": True, "result": "ok"}

        calls = [
            PreparedToolCall(0, "a", "calculator", {"expression": "1"}),
            PreparedToolCall(1, "b", "get_current_time", {}),
        ]
        results = execute_tools_bounded(calls, execute_fn=route, max_concurrency=2)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].get("success"))
        self.assertTrue(results[1].get("success"))

    def test_BD_E09_auth_failure_no_fallback_chain(self):
        names: list[str] = []

        def deny(name, args, **kwargs):
            names.append(name)
            return {
                "success": False,
                "error_code": "TOOL_PERMISSION_DENIED",
                "error": "Tool permission denied",
            }

        result = run_tool_with_reliability(
            "search_code",
            {"query": "x"},
            execute_fn=deny,
            permitted_tools={"search_code", "read_file"},
            registered_tools={"search_code", "read_file"},
        )
        self.assertEqual(names, ["search_code"])
        self.assertFalse(result["reliability"].get("fallback_used"))

    def test_BD_E10_late_timeout_not_claimed_success(self):
        """Timeout path reports failure; does not invent success."""
        from app.services.tool_reliability import _timeout_result

        timed = _timeout_result(
            "calculator",
            worker_may_still_run=True,
            attempts=1,
        )
        self.assertFalse(timed.get("success"))
        self.assertTrue(timed.get("timed_out"))
        self.assertTrue(timed.get("worker_may_still_run"))


if __name__ == "__main__":
    unittest.main()
