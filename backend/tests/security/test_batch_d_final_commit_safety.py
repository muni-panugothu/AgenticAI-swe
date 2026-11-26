"""Batch D-Final — late-worker commit safety (deterministic races)."""

from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from app.chatbot import execute_tool  # noqa: E402
from app.services.execution_lifecycle import (  # noqa: E402
    activate_execution,
    bind_execution,
    execution_public_state,
    finish_commit_success,
    invalidate_execution,
    requires_commit_guard,
    reserve_execution,
    reset_execution_lifecycle_for_tests,
    try_begin_commit,
    unbind_execution,
)
from app.services.request_cancellation import CancellationToken  # noqa: E402
from app.services.tool_execution import reset_global_tool_pool_for_tests  # noqa: E402
from app.services.tool_reliability import (  # noqa: E402
    invoke_sync_once_with_timeout,
    reset_reliability_state_for_tests,
)
from tests.security._fixtures import SecurityCase  # noqa: E402


class TestCommitSafety(SecurityCase):
    def setUp(self):
        super().setUp()
        reset_execution_lifecycle_for_tests()
        reset_global_tool_pool_for_tests()
        reset_reliability_state_for_tests()

    def test_BD_CS00_tool_classification(self):
        self.assertTrue(requires_commit_guard("save_memory"))
        self.assertTrue(requires_commit_guard("forget_memory"))
        self.assertTrue(requires_commit_guard("confirm_memory_update"))
        self.assertFalse(requires_commit_guard("calculator"))
        self.assertFalse(requires_commit_guard("read_file"))
        self.assertFalse(requires_commit_guard("get_current_time"))

    def test_BD_CS01_timeout_before_commit_rejects_side_effect(self):
        eid = reserve_execution(
            tool_name="save_memory",
            approval_id="appr-1",
            user_id="user-a",
        )
        self.assertTrue(activate_execution(eid, approval_id="appr-1"))
        tokens = bind_execution(eid)
        commits = {"n": 0}
        try:
            invalidate_execution(eid, reason="timeout")
            self.assertFalse(try_begin_commit())
            self.assertEqual(commits["n"], 0)
            state = execution_public_state(eid)
            self.assertEqual(state["state"], "invalidated")
            self.assertEqual(state["commit_count"], 0)
        finally:
            unbind_execution(tokens)

    def test_BD_CS02_cancel_before_commit_rejects(self):
        eid = reserve_execution(tool_name="forget_memory", user_id="user-a")
        activate_execution(eid, tool_name="forget_memory")
        token = CancellationToken(request_id="bd-cs02")
        tokens = bind_execution(eid, cancellation=token)
        try:
            token.cancel("client_disconnect")
            self.assertFalse(try_begin_commit())
            self.assertEqual(execution_public_state(eid)["state"], "invalidated")
            self.assertEqual(execution_public_state(eid)["commit_count"], 0)
        finally:
            unbind_execution(tokens)

    def test_BD_CS03_commit_wins_race_before_cancel(self):
        eid = reserve_execution(tool_name="save_memory", user_id="user-a")
        activate_execution(eid, tool_name="save_memory")
        token = CancellationToken(request_id="bd-cs03")
        tokens = bind_execution(eid, cancellation=token)
        try:
            self.assertTrue(try_begin_commit())
            token.cancel("client_disconnect")
            finish_commit_success()
            self.assertFalse(invalidate_execution(eid, reason="cancelled"))
            state = execution_public_state(eid)
            self.assertEqual(state["state"], "committed")
            self.assertEqual(state["commit_count"], 1)
        finally:
            unbind_execution(tokens)

    def test_BD_CS04_timeout_after_commit_does_not_claim_prevented(self):
        eid = reserve_execution(tool_name="save_memory", user_id="user-a")
        activate_execution(eid, tool_name="save_memory")
        tokens = bind_execution(eid)
        try:
            self.assertTrue(try_begin_commit())
            finish_commit_success()
            prevented = invalidate_execution(eid, reason="timeout")
            self.assertFalse(prevented)
            state = execution_public_state(eid)
            self.assertEqual(state["state"], "committed")
            self.assertEqual(state["commit_count"], 1)
        finally:
            unbind_execution(tokens)

    def test_BD_CS05_repeated_cancel_still_blocks_late_commit(self):
        eid = reserve_execution(tool_name="save_memory", user_id="user-a")
        activate_execution(eid, tool_name="save_memory")
        token = CancellationToken(request_id="bd-cs05")
        tokens = bind_execution(eid, cancellation=token)
        try:
            token.cancel("first")
            token.cancel("second")
            self.assertFalse(try_begin_commit())
            self.assertFalse(try_begin_commit())
            self.assertEqual(execution_public_state(eid)["commit_count"], 0)
        finally:
            unbind_execution(tokens)

    def test_BD_CS06_worker_cannot_forge_execution_identity(self):
        forged = "forged-execution-id-never-reserved"
        self.assertFalse(activate_execution(forged, tool_name="save_memory"))
        self.assertIsNone(execution_public_state(forged))

    def test_BD_CS07_cannot_reuse_another_executions_approval(self):
        aid_a, _defaults_a = self.create_and_approve(
            arguments={"memory": "fact a"},
        )
        _aid_b, defaults_b = self.create_and_approve(
            arguments={"memory": "fact b"},
        )
        calls = {"n": 0}

        def _fake(**kwargs):
            calls["n"] += 1
            return {"success": True, "memory_id": "m", "status": "active"}

        with patch.dict("app.chatbot.tools", {"save_memory": _fake}):
            bad = execute_tool(
                "save_memory",
                defaults_b["arguments"],
                user_id="user-a",
                project_id="proj-1",
                tenant_id="ten-1",
                approval_id=aid_a,
            )
        self.assertFalse(bad.get("success"))
        self.assertEqual(calls["n"], 0)

    def test_BD_CS08_end_to_end_timeout_blocks_memory_commit(self):
        aid, defaults = self.create_and_approve(
            arguments={"memory": "late worker fact"},
        )
        commits = {"n": 0}
        entered = threading.Event()
        proceed = threading.Event()
        tool_done = threading.Event()

        def _gated(**kwargs):
            entered.set()
            proceed.wait(timeout=3)
            try:
                if not try_begin_commit():
                    return {
                        "success": False,
                        "error_code": "COMMIT_REJECTED",
                        "mutation_committed": False,
                    }
                commits["n"] += 1
                finish_commit_success()
                return {"success": True, "memory_id": "late", "status": "active"}
            finally:
                tool_done.set()

        result_box: dict = {}
        caller_done = threading.Event()

        def _caller():
            with patch.dict("app.chatbot.tools", {"save_memory": _gated}):
                result_box["timed"] = invoke_sync_once_with_timeout(
                    "save_memory",
                    defaults["arguments"],
                    user_id=defaults["user_id"],
                    project_id=defaults["project_id"],
                    tenant_id=defaults["tenant_id"],
                    approval_id=aid,
                    execute_fn=execute_tool,
                    timeout_seconds=0.05,
                )
            caller_done.set()

        threading.Thread(target=_caller, daemon=True).start()
        self.assertTrue(caller_done.wait(timeout=3))
        # Release worker if it reached the pre-commit pause (or no-op if rejected earlier).
        proceed.set()
        if entered.is_set():
            self.assertTrue(tool_done.wait(timeout=3))

        timed = result_box["timed"]
        self.assertFalse(timed.get("success"))
        self.assertTrue(timed.get("timed_out"))
        self.assertTrue(timed.get("worker_may_still_run"))
        self.assertEqual(commits["n"], 0)

    def test_BD_CS09_legitimate_approved_commit_still_works(self):
        aid, defaults = self.create_and_approve(
            arguments={"memory": "legit durable fact"},
        )
        commits = {"n": 0}

        def _ok(**kwargs):
            if not try_begin_commit():
                return {"success": False, "error_code": "COMMIT_REJECTED"}
            commits["n"] += 1
            finish_commit_success()
            return {"success": True, "memory_id": "m1", "status": "active"}

        with patch.dict("app.chatbot.tools", {"save_memory": _ok}):
            result = execute_tool(
                "save_memory",
                defaults["arguments"],
                user_id=defaults["user_id"],
                project_id=defaults["project_id"],
                tenant_id=defaults["tenant_id"],
                approval_id=aid,
            )
        self.assertTrue(result.get("success"), result)
        self.assertEqual(commits["n"], 1)

    def test_BD_CS10_readonly_tools_unaffected(self):
        result = execute_tool(
            "calculator",
            {"expression": "2+3"},
            user_id="user-a",
        )
        self.assertTrue(result.get("success"), result)

    def test_BD_CS11_barrier_timeout_then_commit_rejected(self):
        eid = reserve_execution(tool_name="save_memory")
        activate_execution(eid, tool_name="save_memory")
        barrier = threading.Barrier(2)
        commits = {"n": 0}
        outcomes: list[bool] = []

        def worker():
            tokens = bind_execution(eid)
            try:
                barrier.wait(timeout=2)
                outcomes.append(try_begin_commit())
                if outcomes[-1]:
                    commits["n"] += 1
                    finish_commit_success()
            finally:
                unbind_execution(tokens)

        t = threading.Thread(target=worker)
        t.start()
        barrier.wait(timeout=2)
        invalidate_execution(eid, reason="timeout")
        t.join(timeout=2)
        self.assertEqual(outcomes, [False])
        self.assertEqual(commits["n"], 0)


if __name__ == "__main__":
    unittest.main()
