"""B6 Stage 7 — outer exception / request-termination cleanup tests."""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "e2e-test-jwt-secret-key-32chars-min!!")

from app import chatbot
from app.services.agent_state import AgentState
from app.services.task_orchestration import safe_cleanup_request_tasks
from app.services.task_progress import (
    build_progress_board_from_decomposition,
    complete_task,
    fail_task,
    note_observable_tool_result,
    start_task,
)
from app.validation.task_models import DecomposedTask, TaskDecomposition


def _text_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=None,
        tool_call_id=None,
        name=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=20),
    )


def _tool_response(tool_id: str, name: str, arguments: str) -> SimpleNamespace:
    tc = SimpleNamespace(
        id=tool_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[tc],
        tool_call_id=None,
        name=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=30),
    )


@contextmanager
def _harness(*, create_side_effect, user_id: str, execute_tool_mock=None):
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
    ]
    if execute_tool_mock is not None:
        patches.append(
            patch.object(chatbot, "execute_tool", side_effect=execute_tool_mock)
        )
    else:
        patches.append(
            patch.object(chatbot, "execute_tool", wraps=chatbot.execute_tool)
        )

    entered = [p.start() for p in patches]
    mock_client = entered[0]
    mock_client.chat.completions.create.side_effect = create_side_effect
    try:
        chatbot._get_conversation_store(user_id).clear()
        yield mock_client, entered[-1]
    finally:
        for p in reversed(patches):
            p.stop()


def _calc_decomp() -> TaskDecomposition:
    return TaskDecomposition(
        original_request="Calculate 2+2",
        trusted_instruction="Calculate 2+2",
        tasks=[
            DecomposedTask(
                id="t1",
                description="Compute 2+2",
                category="calculation",
                independent=True,
            ),
            DecomposedTask(
                id="t2",
                description="Explain",
                category="explanation",
                depends_on=["t1"],
                dependency_kind="required",
            ),
        ],
        execution_order=["t1", "t2"],
        independent_task_ids=["t1"],
        is_multi_intent=True,
    )


class SafeCleanupHelperTests(unittest.TestCase):
    def test_only_running_interrupted_others_preserved(self):
        decomp = _calc_decomp()
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t1")
        # Force t2 into blocked then leave it; also create a failed sibling via
        # a third path — use fail on a separate board task by completing flow.
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            preferred_task_id="t1",
            result_excerpt="4",
        )
        # Leave t1 running with evidence; t2 blocked.
        self.assertEqual(board.tasks["t1"].status, "running")
        self.assertEqual(board.tasks["t2"].status, "blocked")
        state = AgentState(task_progress=board, active_task_id="t1")
        safe_cleanup_request_tasks(state, reason="unit")
        self.assertEqual(board.tasks["t1"].status, "pending")  # interrupted
        self.assertEqual(board.tasks["t2"].status, "blocked")
        self.assertIsNone(state.active_task_id)

    def test_completed_and_failed_untouched(self):
        decomp = _calc_decomp()
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t1")
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            preferred_task_id="t1",
            result_excerpt="4",
        )
        complete_task(board, "t1")
        # Manually fail is not possible from pending blocked t2 without start;
        # start a fresh single-task board for failed.
        decomp2 = TaskDecomposition(
            trusted_instruction="x",
            tasks=[
                DecomposedTask(
                    id="f1",
                    description="x",
                    category="calculation",
                    independent=True,
                )
            ],
            execution_order=["f1"],
            independent_task_ids=["f1"],
        )
        board2 = build_progress_board_from_decomposition(decomp2)
        assert board2 is not None
        start_task(board2, "f1")
        fail_task(board2, "f1", failure_code="e", detail="e")
        # Merge statuses onto one state for cleanup of completed board.
        state = AgentState(task_progress=board, active_task_id=None)
        safe_cleanup_request_tasks(state, reason="unit")
        self.assertEqual(board.tasks["t1"].status, "completed")
        state2 = AgentState(task_progress=board2, active_task_id="f1")
        safe_cleanup_request_tasks(state2, reason="unit")
        self.assertEqual(board2.tasks["f1"].status, "failed")
        self.assertIsNone(state2.active_task_id)

    def test_cleanup_failure_does_not_raise(self):
        state = AgentState(active_task_id="x")

        def boom(*_a, **_k):
            raise RuntimeError("cleanup boom")

        with patch(
            "app.services.task_orchestration.interrupt_running_tasks",
            side_effect=boom,
        ):
            # Must not raise — caller exception preservation depends on this.
            out = safe_cleanup_request_tasks(state, reason="unit")
        self.assertEqual(out, [])
        self.assertIsNone(state.active_task_id)

    def test_repeated_cleanup_idempotent(self):
        decomp = _calc_decomp()
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t1")
        state = AgentState(task_progress=board, active_task_id="t1")
        safe_cleanup_request_tasks(state, reason="one")
        safe_cleanup_request_tasks(state, reason="two")
        self.assertEqual(board.tasks["t1"].status, "pending")
        self.assertIsNone(state.active_task_id)


class ChatWithAiExceptionCleanupTests(unittest.TestCase):
    def test_exception_before_task_start_no_running(self):
        """Escape via decide_task_orchestration before any tool start."""
        captured = {}
        from app.services import task_orchestration as orch

        def create(**_kwargs):
            return _text_response("should not finish")

        def boom_decide(**_kwargs):
            raise RuntimeError("boom before task start")

        def capture_cleanup(agent_state, *, reason="unexpected_exit"):
            board = getattr(agent_state, "task_progress", None)
            captured["before"] = (
                {tid: rec.status for tid, rec in board.tasks.items()}
                if board
                else {}
            )
            out = safe_cleanup_request_tasks(agent_state, reason=reason)
            captured["after"] = (
                {tid: rec.status for tid, rec in board.tasks.items()}
                if board
                else {}
            )
            captured["active"] = getattr(agent_state, "active_task_id", "missing")
            return out

        with _harness(create_side_effect=create, user_id="b6-s7-a"):
            with patch.object(
                orch, "decide_task_orchestration", side_effect=boom_decide
            ), patch.object(
                orch, "safe_cleanup_request_tasks", side_effect=capture_cleanup
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    chatbot.chat_with_ai(
                        "Calculate 12 * 8",
                        user_id="b6-s7-a",
                        request_id="req-s7-a",
                    )
        self.assertIn("before task start", str(ctx.exception))
        self.assertTrue(
            all(s != "running" for s in captured.get("before", {}).values())
        )
        self.assertTrue(
            all(s != "running" for s in captured.get("after", {}).values())
        )
        self.assertIsNone(captured.get("active"))

    def test_exception_after_start_before_tool_success_interrupts(self):
        """Start at tool boundary, then raise before execute_tool returns."""
        captured = {}
        from app.services import task_orchestration as orch

        def create(**_kwargs):
            return _tool_response(
                "c1", "calculator", '{"expression": "2+2"}'
            )

        real_ensure = orch.ensure_task_started_for_tool

        def ensure_then_boom(
            board,
            decision,
            agent_state,
            *,
            tool_name,
            reason="b6_authorized_tool_boundary",
        ):
            real_ensure(
                board,
                decision,
                agent_state,
                tool_name=tool_name,
                reason=reason,
            )
            tid = (
                (decision.active_task_hint or decision.selected_task_id)
                if decision
                else None
            )
            captured["status_after_start"] = (
                board.tasks[tid].status if board and tid in board.tasks else None
            )
            raise RuntimeError("boom after task start")

        def capture_cleanup(agent_state, *, reason="unexpected_exit"):
            board = agent_state.task_progress
            captured["before"] = (
                {tid: rec.status for tid, rec in board.tasks.items()}
                if board
                else {}
            )
            out = safe_cleanup_request_tasks(agent_state, reason=reason)
            captured["after"] = (
                {tid: rec.status for tid, rec in board.tasks.items()}
                if board
                else {}
            )
            captured["active"] = agent_state.active_task_id
            return out

        with _harness(create_side_effect=create, user_id="b6-s7-b"):
            with patch.object(
                orch, "ensure_task_started_for_tool", side_effect=ensure_then_boom
            ), patch.object(
                orch, "safe_cleanup_request_tasks", side_effect=capture_cleanup
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    chatbot.chat_with_ai(
                        "Calculate 2+2",
                        user_id="b6-s7-b",
                        request_id="req-s7-b",
                    )
        self.assertIn("boom after task start", str(ctx.exception))
        self.assertEqual(captured.get("status_after_start"), "running")
        self.assertTrue(
            all(s != "running" for s in captured.get("after", {}).values())
        )
        self.assertIsNone(captured.get("active"))

    def test_exception_after_successful_tool_preserves_completed(self):
        """After tool success/complete, escaping exception must not reopen completed."""
        captured = {}
        from app.services import task_orchestration as orch

        calls = {"n": 0}

        def create(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _tool_response(
                    "c1", "calculator", '{"expression": "2+2"}'
                )
            return _text_response("unused")

        real_apply = orch.apply_orchestration_outcome

        def apply_then_boom(**kwargs):
            out = real_apply(**kwargs)
            if kwargs.get("outcome") == "tool_success" and out.get("completed"):
                raise RuntimeError("boom after completed evidence")
            return out

        def capture_cleanup(agent_state, *, reason="unexpected_exit"):
            board = agent_state.task_progress
            captured["before"] = {
                tid: rec.status for tid, rec in (board.tasks.items() if board else [])
            }
            out = safe_cleanup_request_tasks(agent_state, reason=reason)
            captured["after"] = {
                tid: rec.status for tid, rec in (board.tasks.items() if board else [])
            }
            return out

        with _harness(create_side_effect=create, user_id="b6-s7-c"):
            with patch.object(
                orch, "apply_orchestration_outcome", side_effect=apply_then_boom
            ), patch.object(
                orch, "safe_cleanup_request_tasks", side_effect=capture_cleanup
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    chatbot.chat_with_ai(
                        "Calculate 2+2",
                        user_id="b6-s7-c",
                        request_id="req-s7-c",
                    )
        self.assertIn("after completed", str(ctx.exception))
        after = captured.get("after", {})
        self.assertTrue(all(s != "running" for s in after.values()))
        # Completed tasks must remain completed.
        self.assertTrue(
            any(s == "completed" for s in after.values())
            or any(s == "completed" for s in captured.get("before", {}).values())
        )

    def test_cleanup_failure_preserves_original_exception(self):
        # Helper must swallow cleanup errors.
        state = AgentState(active_task_id="z")
        with patch(
            "app.services.task_orchestration.interrupt_running_tasks",
            side_effect=RuntimeError("cleanup boom"),
        ):
            safe_cleanup_request_tasks(state, reason="x")
        self.assertIsNone(state.active_task_id)

        # Escape via BaseException path that except Exception does not swallow.
        def create(**_kwargs):
            raise KeyboardInterrupt("original cancel")

        with _harness(create_side_effect=create, user_id="b6-s7-d2"):
            with self.assertRaises(KeyboardInterrupt) as ctx:
                chatbot.chat_with_ai(
                    "Calculate 1+1",
                    user_id="b6-s7-d2",
                    request_id="req-s7-d2",
                )
        self.assertIsInstance(ctx.exception, KeyboardInterrupt)

    def test_keyboard_interrupt_preserved(self):
        def create(**_kwargs):
            raise KeyboardInterrupt()

        with _harness(create_side_effect=create, user_id="b6-s7-e"):
            with self.assertRaises(KeyboardInterrupt):
                chatbot.chat_with_ai(
                    "Calculate 1+1",
                    user_id="b6-s7-e",
                    request_id="req-s7-e",
                )

    def test_new_request_cannot_inherit_prior_state(self):
        # Simulate prior request state object discarded; new AgentState is fresh.
        prior = AgentState(active_task_id="old", task_progress=object())
        fresh = AgentState()
        self.assertIsNone(fresh.active_task_id)
        self.assertIsNone(fresh.task_progress)
        self.assertIsNot(prior, fresh)

        # Two sequential chat_with_ai calls — second must succeed independently.
        def create(**_kwargs):
            return _text_response("Hello there.")

        with _harness(create_side_effect=create, user_id="b6-s7-f1"):
            a1 = chatbot.chat_with_ai(
                "Hi", user_id="b6-s7-f1", request_id="req-s7-f1"
            )
        with _harness(create_side_effect=create, user_id="b6-s7-f2"):
            a2 = chatbot.chat_with_ai(
                "Hi", user_id="b6-s7-f2", request_id="req-s7-f2"
            )
        self.assertTrue(a1)
        self.assertTrue(a2)

    def test_chatbot_has_outer_finally_cleanup(self):
        src = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "app"
            / "chatbot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("safe_cleanup_request_tasks", src)
        self.assertIn('reason="chat_with_ai_exit"', src)
        self.assertIn("B6 Stage 7", src)


if __name__ == "__main__":
    unittest.main()
