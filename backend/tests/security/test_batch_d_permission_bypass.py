"""BD-B — Tool permission bypass attempts against prepare/execute boundary."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from app.chatbot import execute_tool, prepare_tool_invocation, tools  # noqa: E402
from app.services.tool_permission_service import (  # noqa: E402
    check_tool_permission,
    grant_tool_permission,
)
from app.services.tool_reliability import (  # noqa: E402
    ToolFailureCategory,
    run_tool_with_reliability,
    select_fallback_tool,
)
from app.validation.agent_security_models import PermissionEffect  # noqa: E402
from tests.security._fixtures import SecurityCase  # noqa: E402


class TestPermissionBypass(SecurityCase):
    def test_BD_B01_call_without_grant_denied(self):
        calls = {"n": 0}

        def _fake(**kwargs):
            calls["n"] += 1
            return {"success": True}

        with patch.dict("app.chatbot.tools", {"drop_database": _fake}):
            # Unregistered tools fail closed before body.
            result = execute_tool("drop_database", {}, user_id="user-a")
        self.assertFalse(result.get("success"))
        self.assertEqual(calls["n"], 0)

    def test_BD_B02_explicit_deny_blocks_baseline_tool(self):
        os.environ["AGENT_PERMISSION_ADMIN_USER_IDS"] = "admin-1"
        grant_tool_permission(
            actor_user_id="admin-1",
            target_user_id="user-a",
            tool_name="calculator",
            effect=PermissionEffect.DENY,
        )
        calls = {"n": 0}

        def _spy(expression):
            calls["n"] += 1
            return {"success": True, "result": "2"}

        with patch.dict("app.chatbot.tools", {"calculator": _spy}):
            result = execute_tool(
                "calculator",
                {"expression": "1+1"},
                user_id="user-a",
            )
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error_code"), "TOOL_PERMISSION_DENIED")
        self.assertEqual(calls["n"], 0)

    def test_BD_B03_legitimate_baseline_still_allowed(self):
        result = execute_tool(
            "calculator",
            {"expression": "2+2"},
            user_id="user-a",
        )
        self.assertTrue(result.get("success"), result)

    def test_BD_B04_fallback_to_denied_tool_rejected(self):
        fb, reason = select_fallback_tool(
            "search_code",
            ToolFailureCategory.TRANSIENT,
            permitted_tools={"search_code"},  # read_file not permitted
            registered_tools={"search_code", "read_file"},
            visited=["search_code"],
            depth=0,
        )
        self.assertIsNone(fb)
        self.assertEqual(reason, "fallback_not_permitted")

    def test_BD_B05_auth_failure_not_retried_via_reliability(self):
        attempts = {"n": 0}

        def deny(name, args, **kwargs):
            attempts["n"] += 1
            return {
                "success": False,
                "error_code": "TOOL_PERMISSION_DENIED",
                "error": "Tool permission denied",
                "mutation_committed": False,
            }

        result = run_tool_with_reliability(
            "search_memory",
            {"query": "x"},
            user_id="user-a",
            execute_fn=deny,
            permitted_tools={"search_memory", "search_memory_summaries"},
            registered_tools={"search_memory", "search_memory_summaries"},
        )
        self.assertEqual(attempts["n"], 1)
        self.assertFalse(result.get("success"))
        self.assertEqual(
            result["reliability"]["failure_category"],
            ToolFailureCategory.AUTHORIZATION.value,
        )
        self.assertFalse(result["reliability"].get("fallback_used"))

    def test_BD_B06_stale_grant_revoked_before_execution(self):
        os.environ["AGENT_PERMISSION_ADMIN_USER_IDS"] = "admin-1"
        grant_tool_permission(
            actor_user_id="admin-1",
            target_user_id="user-a",
            tool_name="calculator",
            effect=PermissionEffect.ALLOW,
        )
        grant_tool_permission(
            actor_user_id="admin-1",
            target_user_id="user-a",
            tool_name="calculator",
            effect=PermissionEffect.DENY,
        )
        result = execute_tool(
            "calculator",
            {"expression": "1+1"},
            user_id="user-a",
        )
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error_code"), "TOOL_PERMISSION_DENIED")

    def test_BD_B07_model_permission_fields_do_not_escalate(self):
        result = execute_tool(
            "calculator",
            {
                "expression": "1+1",
                "permissions": ["*"],
                "role": "admin",
                "grant_id": "g1",
            },
            user_id="user-a",
        )
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error_code"), "UNTRUSTED_SECURITY_FIELD")

    def test_BD_B08_prepare_and_execute_share_boundary(self):
        err, fn, kwargs = prepare_tool_invocation(
            "calculator",
            {"expression": "3+3"},
            user_id="user-a",
        )
        self.assertIsNone(err)
        self.assertIsNotNone(fn)
        self.assertIn("calculator", tools)

    def test_BD_B09_direct_registry_bypass_not_used_by_execute_tool(self):
        """execute_tool must go through prepare; denied tools never invoke body."""
        os.environ["AGENT_PERMISSION_ADMIN_USER_IDS"] = "admin-1"
        grant_tool_permission(
            actor_user_id="admin-1",
            target_user_id="user-a",
            tool_name="get_current_time",
            effect=PermissionEffect.DENY,
        )
        calls = {"n": 0}
        original = tools.get("get_current_time")

        def wrapped(*a, **k):
            calls["n"] += 1
            return original(*a, **k) if original else {"success": True}

        with patch.dict("app.chatbot.tools", {"get_current_time": wrapped}):
            result = execute_tool("get_current_time", {}, user_id="user-a")
        self.assertFalse(result.get("success"))
        self.assertEqual(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
