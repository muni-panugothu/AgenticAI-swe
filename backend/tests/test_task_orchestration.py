"""Phase B6 Stage 4 — task lifecycle and evidence completion tests."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.agent_planning import (
    ACTION_CALL_LLM,
    ACTION_FINALIZE,
    ACTION_FORCE_CONTINUATION,
    ACTION_FORCE_TARGET_READ,
    PlanDecision,
)
from app.services.agent_state import AgentState
from app.services.task_orchestration import (
    append_incomplete_workflow_note,
    apply_orchestration_outcome,
    bind_active_task_hint,
    decide_task_orchestration,
    ensure_task_started_for_tool,
    format_incomplete_workflow_note,
    interrupt_running_tasks,
    observe_orchestration_outcome,
    select_candidate_task_id,
    try_complete_active_task,
)
from app.services.task_progress import (
    TaskProgressError,
    build_progress_board_from_decomposition,
    complete_task,
    eligible_task_ids,
    fail_task,
    note_observable_tool_result,
    record_evidence,
    retry_task,
    set_uncertain_dependency_policy,
    start_task,
)
from app.validation.task_models import DecomposedTask, TaskDecomposition
from app.validation.task_orchestration_models import OrchestrationDecision
from app.validation.task_progress_models import EvidenceRecord


def _decomp(
    tasks: list[DecomposedTask],
    *,
    order: list[str] | None = None,
    needs_clarification: bool = False,
) -> TaskDecomposition:
    ids = [t.id for t in tasks]
    return TaskDecomposition(
        original_request="req",
        trusted_instruction="req",
        is_multi_intent=len(tasks) > 1,
        tasks=tasks,
        execution_order=order or ids,
        independent_task_ids=[t.id for t in tasks if t.independent],
        needs_clarification=needs_clarification,
        clarification_prompts=(
            ["Please clarify the request"] if needs_clarification else []
        ),
    )


def _calc_board() -> tuple[TaskDecomposition, object]:
    decomp = _decomp(
        [
            DecomposedTask(
                id="t1",
                description="Compute 2+2",
                category="calculation",
                independent=True,
            ),
            DecomposedTask(
                id="t2",
                description="Explain the result",
                category="explanation",
                depends_on=["t1"],
                dependency_kind="required",
            ),
        ]
    )
    board = build_progress_board_from_decomposition(decomp)
    assert board is not None
    return decomp, board


def _debug_board() -> tuple[TaskDecomposition, object]:
    decomp = _decomp(
        [
            DecomposedTask(
                id="d1",
                description="Find the auth bug",
                category="debugging",
                independent=True,
            )
        ]
    )
    board = build_progress_board_from_decomposition(decomp)
    assert board is not None
    return decomp, board


def _call_llm_decision(decomp, board, **kwargs):
    return decide_task_orchestration(
        board=board,
        plan_decision=PlanDecision(action=ACTION_CALL_LLM),
        decomposition=decomp,
        tools_enabled=kwargs.pop("tools_enabled", True),
        authorized_tool_names=kwargs.pop(
            "authorized_tool_names",
            ["calculator", "search_code", "read_file", "web_search"],
        ),
        **kwargs,
    )


class SelectionAndPlanTests(unittest.TestCase):
    def test_execution_order_and_single_candidate(self):
        decomp = _decomp(
            [
                DecomposedTask(
                    id="a", description="second", category="explanation", independent=True
                ),
                DecomposedTask(
                    id="b", description="first", category="explanation", independent=True
                ),
            ],
            order=["b", "a"],
        )
        board = build_progress_board_from_decomposition(decomp)
        tid, eligible, _ = select_candidate_task_id(board, decomposition=decomp)
        self.assertEqual(tid, "b")
        self.assertEqual(eligible[0], "b")

    def test_required_optional_uncertain(self):
        decomp, board = _calc_board()
        self.assertEqual(eligible_task_ids(board), ["t1"])
        opt = _decomp(
            [
                DecomposedTask(
                    id="t1", description="a", category="calculation", independent=True
                ),
                DecomposedTask(
                    id="t2",
                    description="b",
                    category="explanation",
                    depends_on=["t1"],
                    dependency_kind="optional",
                ),
            ]
        )
        ob = build_progress_board_from_decomposition(opt)
        self.assertIn("t2", eligible_task_ids(ob))
        unc = _decomp(
            [
                DecomposedTask(
                    id="t1", description="a", category="calculation", independent=True
                ),
                DecomposedTask(
                    id="t2",
                    description="b",
                    category="explanation",
                    depends_on=["t1"],
                    dependency_kind="uncertain",
                ),
            ]
        )
        ub = build_progress_board_from_decomposition(unc)
        set_uncertain_dependency_policy(ub, "block", reason="keep")
        self.assertNotIn("t2", eligible_task_ids(ub))

    def test_finalize_force_abort_budget(self):
        decomp, board = _calc_board()
        fin = decide_task_orchestration(
            board=board,
            plan_decision=PlanDecision(action=ACTION_FINALIZE),
            decomposition=decomp,
            tools_enabled=True,
        )
        self.assertFalse(fin.execution_may_proceed)
        self.assertTrue(fin.partial_finalize_recommended)
        for action in (ACTION_FORCE_CONTINUATION, ACTION_FORCE_TARGET_READ):
            d = decide_task_orchestration(
                board=board,
                plan_decision=PlanDecision(action=action),
                decomposition=decomp,
                tools_enabled=True,
            )
            self.assertFalse(d.execution_may_proceed)
        abort = decide_task_orchestration(
            board=board,
            plan_decision=PlanDecision(action=ACTION_CALL_LLM),
            decomposition=decomp,
            tools_enabled=True,
            stop_action="ABORT",
        )
        self.assertEqual(abort.status, "abort")
        bud = decide_task_orchestration(
            board=board,
            plan_decision=PlanDecision(action=ACTION_CALL_LLM),
            decomposition=decomp,
            tools_enabled=True,
            step=4,
            max_steps=5,
        )
        self.assertEqual(bud.status, "budget_exhausted")


class DeferredStartLifecycleTests(unittest.TestCase):
    def test_call_llm_does_not_start_task(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        self.assertTrue(decision.execution_may_proceed)
        self.assertEqual(board.tasks["t1"].status, "pending")
        state = AgentState()
        bind_active_task_hint(state, decision)
        self.assertEqual(state.active_task_id, "t1")
        self.assertEqual(board.tasks["t1"].status, "pending")

    def test_llm_text_without_tool_leaves_no_running(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        bind_active_task_hint(state, decision)
        # Simulate LLM ordinary text stop — never started.
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="llm_text_stop",
        )
        self.assertEqual(board.tasks["t1"].status, "pending")
        self.assertIsNone(state.active_task_id)

    def test_llm_failure_before_execution_no_orphan(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        bind_active_task_hint(state, decision)
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="unknown",
        )
        self.assertNotEqual(board.tasks["t1"].status, "running")
        self.assertIsNone(state.active_task_id)

    def test_start_only_at_compatible_tool_boundary(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        bind_active_task_hint(state, decision)
        # Unrelated tool must not start the calculation task.
        self.assertFalse(
            ensure_task_started_for_tool(
                board, decision, state, tool_name="web_search"
            )
        )
        self.assertEqual(board.tasks["t1"].status, "pending")
        self.assertTrue(
            ensure_task_started_for_tool(
                board, decision, state, tool_name="calculator"
            )
        )
        self.assertEqual(board.tasks["t1"].status, "running")
        self.assertEqual(state.active_task_id, "t1")

    def test_invalid_args_after_start_interrupts_empty(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="validation_error",
            tool_success=False,
            active_task_id="t1",
        )
        self.assertEqual(board.tasks["t1"].status, "pending")  # interrupted
        self.assertIsNone(state.active_task_id)

    def test_tool_failure_no_success_evidence(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=False,
            preferred_task_id="t1",
        )
        self.assertEqual(board.tasks["t1"].evidence, [])
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="tool_error",
            tool_success=False,
            active_task_id="t1",
        )
        self.assertEqual(board.tasks["t1"].status, "pending")


class EvidenceCompletionTests(unittest.TestCase):
    def test_calculator_success_completes(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            preferred_task_id="t1",
            result_excerpt="4",
        )
        out = apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="tool_success",
            tool_success=True,
            active_task_id="t1",
        )
        self.assertTrue(out["completed"])
        self.assertEqual(board.tasks["t1"].status, "completed")
        self.assertIsNone(state.active_task_id)
        # Successor becomes eligible after completion.
        self.assertIn("t2", eligible_task_ids(board))

    def test_debugging_search_only_incomplete(self):
        decomp, board = _debug_board()
        decision = _call_llm_decision(
            decomp,
            board,
            authorized_tool_names=["search_code", "read_file"],
        )
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="search_code"
        )
        note_observable_tool_result(
            board,
            tool_name="search_code",
            success=True,
            preferred_task_id="d1",
            result_excerpt="hits",
        )
        out = apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="tool_success",
            tool_success=True,
            active_task_id="d1",
        )
        self.assertFalse(out["completed"])
        self.assertEqual(board.tasks["d1"].status, "running")

    def test_debugging_search_plus_read_completes(self):
        decomp, board = _debug_board()
        decision = _call_llm_decision(
            decomp,
            board,
            authorized_tool_names=["search_code", "read_file"],
        )
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="search_code"
        )
        note_observable_tool_result(
            board,
            tool_name="search_code",
            success=True,
            preferred_task_id="d1",
            result_excerpt="hits",
        )
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="tool_success",
            tool_success=True,
            active_task_id="d1",
        )
        self.assertEqual(board.tasks["d1"].status, "running")
        # Second tool call for same running task.
        ensure_task_started_for_tool(
            board, decision, state, tool_name="read_file"
        )
        note_observable_tool_result(
            board,
            tool_name="read_file",
            success=True,
            preferred_task_id="d1",
            result_excerpt="def login",
        )
        out = apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="tool_success",
            tool_success=True,
            active_task_id="d1",
        )
        self.assertTrue(out["completed"])
        self.assertEqual(board.tasks["d1"].status, "completed")
        self.assertEqual(len(board.tasks["d1"].evidence), 2)

    def test_unrelated_tool_cannot_complete_or_attach(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        # Preferred set but web_search incompatible → unassigned, not on t1.
        note_observable_tool_result(
            board,
            tool_name="web_search",
            success=True,
            preferred_task_id="t1",
            result_excerpt="news",
        )
        self.assertEqual(board.tasks["t1"].evidence, [])
        self.assertEqual(len(board.unassigned_evidence), 1)
        self.assertFalse(try_complete_active_task(board, "t1", state))
        self.assertEqual(board.tasks["t1"].status, "running")

    def test_retry_stale_evidence_rejected(self):
        decomp, board = _calc_board()
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="calculator_result",
                source="tool_result",
                summary="4",
                tool_name="calculator",
                verified=True,
                details={"attempt": 1},
            ),
        )
        fail_task(board, "t1", failure_code="x", detail="x")
        retry_task(board, "t1", reason="manual")
        start_task(board, "t1", reason="resume")
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")

    def test_no_repeated_start_bumps_attempt(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        attempt = board.tasks["t1"].attempt_count
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        self.assertEqual(board.tasks["t1"].attempt_count, attempt)
        self.assertEqual(board.tasks["t1"].status, "running")


class StopCleanupAndHonestyTests(unittest.TestCase):
    def test_finalize_interrupts_running_partial_note(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        fin = decide_task_orchestration(
            board=board,
            plan_decision=PlanDecision(action=ACTION_FINALIZE),
            decomposition=decomp,
            tools_enabled=True,
        )
        self.assertTrue(fin.partial_finalize_recommended)
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=fin,
            outcome="finalize",
        )
        self.assertEqual(board.tasks["t1"].status, "pending")  # interrupted
        self.assertNotEqual(board.tasks["t1"].status, "completed")
        self.assertIsNone(state.active_task_id)
        note = format_incomplete_workflow_note(board)
        self.assertIn("not fully finished", note)
        answer = append_incomplete_workflow_note("Done.", board)
        self.assertIn("partial", answer.casefold())

    def test_abort_and_budget_cleanup(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        apply_orchestration_outcome(
            board=board, agent_state=state, decision=decision, outcome="abort"
        )
        self.assertNotEqual(board.tasks["t1"].status, "running")
        ensure_task_started_for_tool(
            board,
            _call_llm_decision(decomp, board),
            state,
            tool_name="calculator",
        )
        apply_orchestration_outcome(
            board=board, agent_state=state, decision=decision, outcome="budget"
        )
        self.assertNotEqual(board.tasks["t1"].status, "running")

    def test_clarification_clears_hint_and_unsupported_no_start(self):
        decomp = _decomp(
            [
                DecomposedTask(
                    id="t1",
                    description="Do the thing",
                    category="explanation",
                    independent=True,
                    needs_clarification=True,
                    clarification_prompt="Which thing?",
                )
            ],
            needs_clarification=True,
        )
        board = build_progress_board_from_decomposition(decomp)
        decision = decide_task_orchestration(
            board=board,
            plan_decision=PlanDecision(action=ACTION_CALL_LLM),
            decomposition=decomp,
            tools_enabled=False,
        )
        self.assertEqual(decision.status, "clarification_required")
        state = AgentState(active_task_id="stale")
        bind_active_task_hint(state, decision)
        self.assertIsNone(state.active_task_id)

        codegen = _decomp(
            [
                DecomposedTask(
                    id="c1",
                    description="Build API",
                    category="api_development",
                    independent=True,
                )
            ]
        )
        cb = build_progress_board_from_decomposition(codegen)
        ud = decide_task_orchestration(
            board=cb,
            plan_decision=PlanDecision(action=ACTION_CALL_LLM),
            decomposition=codegen,
            tools_enabled=True,
            authorized_tool_names=["search_code", "read_file"],
        )
        self.assertEqual(ud.status, "unsupported_path")
        self.assertFalse(ud.execution_may_proceed)
        self.assertFalse(
            ensure_task_started_for_tool(
                cb, ud, AgentState(), tool_name="search_code"
            )
        )
        self.assertEqual(cb.tasks["c1"].status, "pending")

    def test_completed_clears_active_task_id(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            preferred_task_id="t1",
            result_excerpt="4",
        )
        try_complete_active_task(board, "t1", state)
        self.assertIsNone(state.active_task_id)

    def test_cross_request_isolation(self):
        decomp, board = _calc_board()
        s1 = AgentState(task_progress=board, active_task_id="t1")
        s2 = AgentState()
        self.assertIsNone(s2.active_task_id)
        self.assertIsNone(s2.task_progress)
        self.assertIsNot(s1, s2)

    def test_preferred_does_not_broadcast_to_other_running(self):
        decomp = _decomp(
            [
                DecomposedTask(
                    id="calc",
                    description="calc",
                    category="calculation",
                    independent=True,
                ),
                DecomposedTask(
                    id="dbg",
                    description="dbg",
                    category="debugging",
                    independent=True,
                ),
            ]
        )
        board = build_progress_board_from_decomposition(decomp)
        start_task(board, "calc")
        start_task(board, "dbg")
        note_observable_tool_result(
            board,
            tool_name="search_code",
            success=True,
            preferred_task_id="calc",  # calc rejects search → unassigned
            result_excerpt="hits",
        )
        self.assertEqual(board.tasks["calc"].evidence, [])
        self.assertEqual(board.tasks["dbg"].evidence, [])
        self.assertEqual(len(board.unassigned_evidence), 1)

    def test_security_no_execute_tool_in_orchestration_module(self):
        src = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "services"
            / "task_orchestration.py"
        )
        tree = ast.parse(src.read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == "execute_tool")
                or (
                    isinstance(n.func, ast.Attribute)
                    and n.func.attr == "execute_tool"
                )
            )
        ]
        self.assertEqual(calls, [])
        with patch("app.chatbot.execute_tool") as mock_exec:
            decomp, board = _calc_board()
            decide_task_orchestration(
                board=board,
                plan_decision=PlanDecision(action=ACTION_CALL_LLM),
                decomposition=decomp,
                tools_enabled=True,
            )
            mock_exec.assert_not_called()

    def test_observe_guidance_never_marks_success(self):
        d = OrchestrationDecision(
            status="candidate_selected",
            selected_task_id="t1",
            execution_may_proceed=True,
        )
        for kwargs in (
            {"tool_success": True},
            {"tool_success": False},
            {"unknown_outcome": True},
            {"retry_exhausted": True},
        ):
            out = observe_orchestration_outcome(decision=d, **kwargs)
            self.assertFalse(out["treat_as_success"])

    def test_interrupt_running_helper(self):
        decomp, board = _calc_board()
        start_task(board, "t1")
        state = AgentState(active_task_id="t1")
        interrupted = interrupt_running_tasks(
            board, reason="unit_test", agent_state=state
        )
        self.assertEqual(interrupted, ["t1"])
        self.assertEqual(board.tasks["t1"].status, "pending")
        self.assertIsNone(state.active_task_id)


class Stage6HardeningTests(unittest.TestCase):
    """Audit-driven defects C1/C2 and related lifecycle hardening."""

    def test_repeated_failure_style_abort_cleans_running(self):
        """Mirrors chatbot max-same-tool-failure exit cleanup."""
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        self.assertEqual(board.tasks["t1"].status, "running")
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="abort",
        )
        self.assertNotEqual(board.tasks["t1"].status, "running")
        self.assertIsNone(state.active_task_id)
        note = format_incomplete_workflow_note(board)
        self.assertIn("not fully finished", note)

    def test_max_steps_style_budget_cleans_running(self):
        """Mirrors chatbot max-agent-steps exit cleanup."""
        decomp, board = _debug_board()
        decision = _call_llm_decision(
            decomp,
            board,
            authorized_tool_names=["search_code", "read_file"],
        )
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="search_code"
        )
        note_observable_tool_result(
            board,
            tool_name="search_code",
            success=True,
            preferred_task_id="d1",
            result_excerpt="hits",
        )
        # Partial evidence — still running, then budget stop.
        self.assertEqual(board.tasks["d1"].status, "running")
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="budget",
        )
        self.assertEqual(board.tasks["d1"].status, "pending")
        self.assertIsNone(state.active_task_id)

    def test_cleanup_never_interrupts_completed(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            preferred_task_id="t1",
            result_excerpt="4",
        )
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="tool_success",
            tool_success=True,
            active_task_id="t1",
        )
        self.assertEqual(board.tasks["t1"].status, "completed")
        interrupted = interrupt_running_tasks(
            board, reason="finalize_after_complete", agent_state=state
        )
        self.assertEqual(interrupted, [])
        self.assertEqual(board.tasks["t1"].status, "completed")

    def test_incomplete_note_not_duplicated(self):
        decomp, board = _calc_board()
        start_task(board, "t1")
        interrupt_running_tasks(board, reason="stop")
        once = append_incomplete_workflow_note("Answer body.", board)
        twice = append_incomplete_workflow_note(once, board)
        self.assertEqual(once.count("not fully finished"), 1)
        self.assertEqual(twice.count("not fully finished"), 1)
        self.assertEqual(once, twice)

    def test_never_started_board_has_no_incomplete_note(self):
        """Advisory B4 boards that orchestration never entered stay silent."""
        decomp, board = _calc_board()
        self.assertTrue(board.tasks)
        self.assertTrue(
            all(rec.attempt_count == 0 for rec in board.tasks.values())
        )
        note = format_incomplete_workflow_note(board)
        self.assertEqual(note, "")
        self.assertEqual(
            append_incomplete_workflow_note("Hello!", board),
            "Hello!",
        )

    def test_incomplete_note_distinguishes_statuses(self):
        decomp, board = _calc_board()
        start_task(board, "t1")
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            preferred_task_id="t1",
            result_excerpt="4",
        )
        complete_task(board, "t1")
        # t2 unlocks to pending after t1 completes.
        self.assertEqual(board.tasks["t2"].status, "pending")
        note = format_incomplete_workflow_note(board)
        self.assertIn("completed", note)
        self.assertIn("waiting", note)
        self.assertIn("not fully finished", note)

    def test_unauthorized_incompatible_tool_cannot_start(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(
            decomp,
            board,
            authorized_tool_names=["web_search"],  # cannot serve calculation
        )
        # Catalog cannot serve category → unsupported, no start.
        self.assertEqual(decision.status, "unsupported_path")
        self.assertFalse(
            ensure_task_started_for_tool(
                board, decision, AgentState(), tool_name="web_search"
            )
        )
        self.assertEqual(board.tasks["t1"].status, "pending")

    def test_invalid_args_style_no_completion(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=False,
            preferred_task_id="t1",
        )
        self.assertEqual(board.tasks["t1"].evidence, [])
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="validation_error",
            tool_success=False,
            active_task_id="t1",
        )
        self.assertNotEqual(board.tasks["t1"].status, "completed")
        self.assertNotEqual(board.tasks["t1"].status, "running")

    def test_overlapping_categories_no_cross_attribution(self):
        """Two debugging tasks running — preferred must not leak to sibling."""
        decomp = _decomp(
            [
                DecomposedTask(
                    id="d1",
                    description="bug A",
                    category="debugging",
                    independent=True,
                ),
                DecomposedTask(
                    id="d2",
                    description="bug B",
                    category="debugging",
                    independent=True,
                ),
            ]
        )
        board = build_progress_board_from_decomposition(decomp)
        start_task(board, "d1")
        start_task(board, "d2")
        note_observable_tool_result(
            board,
            tool_name="search_code",
            success=True,
            preferred_task_id="d1",
            result_excerpt="A hits",
        )
        self.assertEqual(len(board.tasks["d1"].evidence), 1)
        self.assertEqual(board.tasks["d2"].evidence, [])

    def test_provider_failure_style_finalize_cleans(self):
        decomp, board = _calc_board()
        decision = _call_llm_decision(decomp, board)
        state = AgentState(task_progress=board)
        ensure_task_started_for_tool(
            board, decision, state, tool_name="calculator"
        )
        apply_orchestration_outcome(
            board=board,
            agent_state=state,
            decision=decision,
            outcome="finalize",
        )
        self.assertNotEqual(board.tasks["t1"].status, "running")
        self.assertIsNone(state.active_task_id)

    def test_chatbot_exits_call_b6_close_helper(self):
        """Source-level guard: C1/C2 exits must invoke _b6_close_and_annotate."""
        src = (
            Path(__file__).resolve().parents[1] / "app" / "chatbot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_b6_close_and_annotate", src)
        self.assertIn('outcome="abort"', src)
        self.assertIn('outcome="budget"', src)
        # Max-failure and max-steps sections both annotate.
        self.assertGreaterEqual(src.count("_b6_close_and_annotate"), 3)


if __name__ == "__main__":
    unittest.main()
