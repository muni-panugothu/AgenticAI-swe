"""BD-G — Streaming / parallel execution security."""

from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from app.chatbot import execute_tool  # noqa: E402
from app.services.request_cancellation import (  # noqa: E402
    CancellationToken,
    RequestCancelled,
)
from app.services.tool_execution import (  # noqa: E402
    PreparedToolCall,
    can_parallelize,
    execute_tools_bounded,
    is_parallel_safe,
    partition_execution_batches,
    reset_global_tool_pool_for_tests,
)
from app.services.tool_permission_service import grant_tool_permission  # noqa: E402
from app.validation.agent_security_models import PermissionEffect  # noqa: E402
from tests.security._fixtures import SecurityCase  # noqa: E402


class TestStreamingParallelSecurity(SecurityCase):
    def setUp(self):
        super().setUp()
        reset_global_tool_pool_for_tests()

    def test_BD_G01_unauthorized_tool_in_batch_not_executed(self):
        os.environ["AGENT_PERMISSION_ADMIN_USER_IDS"] = "admin-1"
        grant_tool_permission(
            actor_user_id="admin-1",
            target_user_id="user-a",
            tool_name="get_current_time",
            effect=PermissionEffect.DENY,
        )
        executed: list[str] = []

        def route(name, args, **kwargs):
            executed.append(name)
            return {"success": True, "result": name}

        # Real boundary: execute_tool (not a naked execute_fn) for denied tool.
        denied = execute_tool("get_current_time", {}, user_id="user-a")
        self.assertFalse(denied.get("success"))
        self.assertEqual(denied.get("error_code"), "TOOL_PERMISSION_DENIED")

        allowed = execute_tool(
            "calculator",
            {"expression": "1+1"},
            user_id="user-a",
        )
        self.assertTrue(allowed.get("success"), allowed)
        # Custom fn path still preserves sibling independence.
        calls = [
            PreparedToolCall(0, "a", "calculator", {"expression": "1"}),
            PreparedToolCall(1, "b", "get_current_time", {}),
        ]
        results = execute_tools_bounded(
            calls,
            user_id="user-a",
            execute_fn=route,
            max_concurrency=2,
            permitted_tools=["calculator"],
        )
        self.assertTrue(results[0].get("success"))
        self.assertFalse(results[1].get("success"))
        self.assertEqual(results[1].get("error_code"), "TOOL_NOT_PERMITTED")
        self.assertNotIn("get_current_time", executed)

    def test_BD_G02_only_parallel_safe_readonly_overlap(self):
        self.assertTrue(is_parallel_safe("calculator"))
        self.assertTrue(is_parallel_safe("get_current_time"))
        self.assertFalse(is_parallel_safe("save_memory"))
        self.assertFalse(is_parallel_safe("forget_memory"))
        mixed = [
            PreparedToolCall(0, "a", "calculator", {"expression": "1"}),
            PreparedToolCall(1, "b", "save_memory", {"memory": "x"}),
        ]
        self.assertFalse(can_parallelize(mixed))
        batches = partition_execution_batches(mixed)
        self.assertEqual(len(batches), 2)

    def test_BD_G03_cancel_before_tool_execution(self):
        token = CancellationToken(request_id="bd-g03")
        token.cancel("client_disconnect")
        calls = [
            PreparedToolCall(0, "a", "calculator", {"expression": "1"}),
        ]
        with self.assertRaises(RequestCancelled):
            execute_tools_bounded(
                calls,
                cancellation=token,
                execute_fn=lambda *a, **k: {"success": True},
            )

    def test_BD_G04_duplicate_tool_call_ids_still_authorize_each(self):
        calls = [
            PreparedToolCall(0, "same", "calculator", {"expression": "1+1"}),
            PreparedToolCall(1, "same", "calculator", {"expression": "2+2"}),
        ]
        results = execute_tools_bounded(
            calls,
            user_id="user-a",
            max_concurrency=2,
            execute_fn=lambda name, args, **k: {
                "success": True,
                "result": args.get("expression"),
            },
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.get("success") for r in results))

    def test_BD_G05_mutating_tool_not_parallelized_with_reads(self):
        calls = [
            PreparedToolCall(0, "r1", "calculator", {"expression": "1"}),
            PreparedToolCall(1, "m1", "save_memory", {"memory": "x"}),
            PreparedToolCall(2, "r2", "get_current_time", {}),
        ]
        batches = partition_execution_batches(calls)
        # save_memory must be alone.
        self.assertTrue(any(len(b) == 1 and b[0].tool_name == "save_memory" for b in batches))

    def test_BD_G06_permission_change_between_plan_and_execute(self):
        os.environ["AGENT_PERMISSION_ADMIN_USER_IDS"] = "admin-1"
        # Plan-time would have allowed calculator; revoke before execute.
        grant_tool_permission(
            actor_user_id="admin-1",
            target_user_id="user-a",
            tool_name="calculator",
            effect=PermissionEffect.DENY,
        )
        result = execute_tool(
            "calculator",
            {"expression": "9+9"},
            user_id="user-a",
        )
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error_code"), "TOOL_PERMISSION_DENIED")

    def test_BD_G07_malformed_args_rejected_before_body(self):
        calls = {"n": 0}

        def spy(expression):
            calls["n"] += 1
            return {"success": True, "result": "x"}

        with patch.dict("app.chatbot.tools", {"calculator": spy}):
            result = execute_tool(
                "calculator",
                {"expression": {"nested": True}},
                user_id="user-a",
            )
        self.assertFalse(result.get("success"))
        self.assertEqual(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
