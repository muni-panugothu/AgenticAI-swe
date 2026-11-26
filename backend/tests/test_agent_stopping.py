"""Topic 40 — typed agent-loop stopping façade."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app import chatbot
from app.services.agent_planning import (
    ACTION_CALL_LLM,
    ACTION_CONTINUE_WITH_NUDGE,
    ACTION_FINALIZE,
    ACTION_FORCE_CONTINUATION,
    ACTION_FORCE_TARGET_READ,
    PlanDecision,
    plan_next_action,
)
from app.services.agent_routing import route_request
from app.services.agent_stopping import (
    OUTCOME_FINALIZE_NONE,
    OUTCOME_PROVIDER_ABORT,
    OUTCOME_PROVIDER_DEGRADE,
    OUTCOME_REPEATED_FAILURE,
    STOP_ABORT,
    STOP_CONTINUE_LLM,
    STOP_CONTINUE_NUDGE,
    STOP_FINALIZE_COMPLETE,
    STOP_FINALIZE_PARTIAL,
    STOP_FORCE_CONTINUATION,
    STOP_FORCE_TARGET_READ,
    STOP_REPEATED_FAILURE,
    StopDecision,
    evaluate_loop_stop,
    map_plan_decision,
)


class TestPlanDecisionMapping(unittest.TestCase):
    """Each authoritative PlanDecision maps to the correct StopDecision."""

    def test_call_llm_maps_to_continue_llm(self):
        plan = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=False,
            step=0,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=0,
            tool_budget_total=20,
            uncovered_count=0,
        )
        self.assertEqual(plan.action, ACTION_CALL_LLM)
        stop = evaluate_loop_stop(plan=plan, tools_enabled=True)
        self.assertEqual(stop.action, STOP_CONTINUE_LLM)
        self.assertEqual(stop.plan_action, ACTION_CALL_LLM)
        self.assertNotIn(
            stop.action,
            {STOP_FINALIZE_COMPLETE, STOP_FINALIZE_PARTIAL, STOP_ABORT},
        )

    def test_force_continuation_maps(self):
        plan = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=False,
            step=1,
            max_steps=8,
            pending_continuation=True,
            total_tool_executions=1,
            tool_budget_total=20,
            uncovered_count=0,
            investigation_started=True,
        )
        self.assertEqual(plan.action, ACTION_FORCE_CONTINUATION)
        stop = map_plan_decision(plan)
        self.assertEqual(stop.action, STOP_FORCE_CONTINUATION)
        self.assertEqual(stop.plan_action, ACTION_FORCE_CONTINUATION)

    def test_force_target_read_maps(self):
        plan = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=False,
            step=1,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=1,
            tool_budget_total=20,
            uncovered_count=2,
            investigation_started=True,
            blocked_in_a_row=1,
            successful_reads=1,
        )
        self.assertEqual(plan.action, ACTION_FORCE_TARGET_READ)
        stop = evaluate_loop_stop(plan=plan)
        self.assertEqual(stop.action, STOP_FORCE_TARGET_READ)
        self.assertEqual(stop.plan_action, ACTION_FORCE_TARGET_READ)

    def test_continue_nudge_maps(self):
        plan = plan_next_action(
            phase="post_llm_stop",
            tools_enabled=True,
            step=1,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=2,
            tool_budget_total=14,
            uncovered_count=0,
            premature_answer=True,
            has_evidence_snippets=True,
            requirements_met=False,
            successful_searches=1,
            successful_reads=1,
        )
        self.assertEqual(plan.action, ACTION_CONTINUE_WITH_NUDGE)
        stop = evaluate_loop_stop(plan=plan)
        self.assertEqual(stop.action, STOP_CONTINUE_NUDGE)
        self.assertEqual(stop.plan_action, ACTION_CONTINUE_WITH_NUDGE)

    def test_finalize_complete_answer_ready(self):
        plan = plan_next_action(
            phase="post_llm_stop",
            tools_enabled=True,
            step=3,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=4,
            tool_budget_total=14,
            uncovered_count=0,
            premature_answer=False,
            has_evidence_snippets=True,
            requirements_met=True,
            successful_searches=1,
            successful_reads=1,
        )
        self.assertEqual(plan.action, ACTION_FINALIZE)
        self.assertEqual(plan.reason, "answer_ready")
        stop = evaluate_loop_stop(plan=plan)
        self.assertEqual(stop.action, STOP_FINALIZE_COMPLETE)
        self.assertEqual(stop.plan_action, ACTION_FINALIZE)

    def test_finalize_partial_requirements_incomplete(self):
        plan = plan_next_action(
            phase="post_llm_stop",
            tools_enabled=True,
            step=7,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=10,
            tool_budget_total=14,
            uncovered_count=1,
            premature_answer=True,
            has_evidence_snippets=True,
            requirements_met=False,
            successful_searches=1,
            successful_reads=1,
        )
        self.assertEqual(plan.action, ACTION_FINALIZE)
        self.assertEqual(plan.reason, "requirements_incomplete")
        stop = evaluate_loop_stop(plan=plan)
        self.assertEqual(stop.action, STOP_FINALIZE_PARTIAL)
        self.assertEqual(stop.reason, "requirements_incomplete")
        self.assertEqual(stop.plan_action, ACTION_FINALIZE)

    def test_force_final_finalize_maps_complete(self):
        plan = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=True,
            step=7,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=5,
            tool_budget_total=20,
            uncovered_count=0,
        )
        self.assertEqual(plan.action, ACTION_FINALIZE)
        self.assertEqual(plan.reason, "force_final")
        stop = evaluate_loop_stop(plan=plan)
        self.assertEqual(stop.action, STOP_FINALIZE_COMPLETE)


class TestContinueDoesNotFinalize(unittest.TestCase):
    def test_continue_llm_is_not_a_finalize_action(self):
        plan = PlanDecision(action=ACTION_CALL_LLM, reason="pre_llm_call")
        stop = evaluate_loop_stop(plan=plan, tools_enabled=True)
        self.assertEqual(stop.action, STOP_CONTINUE_LLM)
        self.assertFalse(stop.action.startswith("FINALIZE"))
        self.assertNotEqual(stop.action, STOP_ABORT)

    def test_tools_disabled_stays_continue_llm(self):
        plan = plan_next_action(phase="pre_llm", tools_enabled=False)
        self.assertEqual(plan.action, ACTION_CALL_LLM)
        self.assertEqual(plan.reason, "tools_disabled")
        stop = evaluate_loop_stop(plan=plan, tools_enabled=False)
        self.assertEqual(stop.action, STOP_CONTINUE_LLM)
        self.assertEqual(stop.reason, "tools_disabled_direct")
        self.assertNotIn(
            stop.action,
            {
                STOP_FORCE_CONTINUATION,
                STOP_FORCE_TARGET_READ,
                STOP_CONTINUE_NUDGE,
                STOP_FINALIZE_PARTIAL,
            },
        )


class TestNonPlanOutcomes(unittest.TestCase):
    def test_repeated_tool_failure_outcome(self):
        stop = evaluate_loop_stop(outcome=OUTCOME_REPEATED_FAILURE)
        self.assertEqual(stop.action, STOP_REPEATED_FAILURE)
        self.assertEqual(stop.reason, "max_same_tool_failures")

    def test_finalize_none_outcome(self):
        stop = evaluate_loop_stop(outcome=OUTCOME_FINALIZE_NONE)
        self.assertEqual(stop.action, STOP_FINALIZE_COMPLETE)
        self.assertEqual(stop.reason, "resolve_none")

    def test_provider_degrade_and_abort(self):
        degrade = evaluate_loop_stop(outcome=OUTCOME_PROVIDER_DEGRADE)
        self.assertEqual(degrade.action, STOP_FINALIZE_PARTIAL)
        abort = evaluate_loop_stop(outcome=OUTCOME_PROVIDER_ABORT)
        self.assertEqual(abort.action, STOP_ABORT)

    def test_outcome_wins_over_plan(self):
        plan = PlanDecision(action=ACTION_CALL_LLM, reason="pre_llm_call")
        stop = evaluate_loop_stop(
            plan=plan,
            outcome=OUTCOME_REPEATED_FAILURE,
            tools_enabled=True,
        )
        self.assertEqual(stop.action, STOP_REPEATED_FAILURE)
        self.assertEqual(stop.plan_action, ACTION_CALL_LLM)

    def test_each_recognized_outcome_mapping(self):
        expected = {
            OUTCOME_REPEATED_FAILURE: STOP_REPEATED_FAILURE,
            OUTCOME_FINALIZE_NONE: STOP_FINALIZE_COMPLETE,
            OUTCOME_PROVIDER_DEGRADE: STOP_FINALIZE_PARTIAL,
            OUTCOME_PROVIDER_ABORT: STOP_ABORT,
        }
        for tag, action in expected.items():
            with self.subTest(tag=tag):
                stop = evaluate_loop_stop(outcome=tag)
                self.assertEqual(stop.action, action)


class TestUnknownOutcomeSafety(unittest.TestCase):
    """Unknown outcome must ABORT and must not fall through to plan finalize."""

    def test_unknown_outcome_with_finalize_plan_aborts(self):
        plan = PlanDecision(action=ACTION_FINALIZE, reason="answer_ready")
        stop = evaluate_loop_stop(plan=plan, outcome="typo_outcome")
        self.assertEqual(stop.action, STOP_ABORT)
        self.assertTrue(stop.reason.startswith("unknown_outcome:"))
        self.assertNotIn(
            stop.action,
            {STOP_FINALIZE_COMPLETE, STOP_FINALIZE_PARTIAL},
        )
        self.assertEqual(stop.plan_action, ACTION_FINALIZE)

    def test_unknown_outcome_with_call_llm_plan_aborts(self):
        plan = PlanDecision(action=ACTION_CALL_LLM, reason="pre_llm_call")
        stop = evaluate_loop_stop(plan=plan, outcome="not_a_real_tag")
        self.assertEqual(stop.action, STOP_ABORT)
        self.assertIn("unknown_outcome", stop.reason)
        self.assertEqual(stop.plan_action, ACTION_CALL_LLM)

    def test_unknown_outcome_without_plan_aborts(self):
        stop = evaluate_loop_stop(outcome="garbage", tools_enabled=True)
        self.assertEqual(stop.action, STOP_ABORT)
        self.assertEqual(stop.reason, "unknown_outcome:garbage")
        self.assertIsNone(stop.plan_action)

    def test_no_outcome_with_valid_plan_unchanged(self):
        plan = PlanDecision(action=ACTION_FINALIZE, reason="answer_ready")
        stop = evaluate_loop_stop(plan=plan)
        self.assertEqual(stop.action, STOP_FINALIZE_COMPLETE)
        self.assertEqual(stop.plan_action, ACTION_FINALIZE)

    def test_no_outcome_no_plan_unchanged(self):
        abort = evaluate_loop_stop(tools_enabled=True)
        self.assertEqual(abort.action, STOP_ABORT)
        self.assertEqual(abort.reason, "missing_plan_and_outcome")
        cont = evaluate_loop_stop(tools_enabled=False)
        self.assertEqual(cont.action, STOP_CONTINUE_LLM)
        self.assertEqual(cont.reason, "tools_disabled_no_plan")


class TestPlanningRemainsAuthoritative(unittest.TestCase):
    def test_stop_does_not_override_force_continuation_plan(self):
        plan = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=True,
            step=1,
            max_steps=8,
            pending_continuation=True,
            total_tool_executions=1,
            tool_budget_total=20,
            uncovered_count=0,
        )
        self.assertEqual(plan.action, ACTION_FORCE_CONTINUATION)
        stop = evaluate_loop_stop(plan=plan)
        # Façade names FORCE_CONTINUATION — does not flip to FINALIZE.
        self.assertEqual(stop.action, STOP_FORCE_CONTINUATION)
        self.assertNotEqual(stop.action, STOP_FINALIZE_COMPLETE)

    def test_incomplete_evidence_finalize_stays_partial(self):
        plan = PlanDecision(
            action=ACTION_FINALIZE,
            reason="requirements_incomplete",
        )
        stop = evaluate_loop_stop(plan=plan)
        self.assertEqual(stop.action, STOP_FINALIZE_PARTIAL)
        self.assertEqual(stop.plan_action, ACTION_FINALIZE)


class TestRagAndToolsDisabledPaths(unittest.TestCase):
    def test_rag_route_does_not_enter_force_stop_matrix(self):
        route = route_request(
            user_message="What does the document say?",
            rag_context="Evidence about RAG.",
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertFalse(route.tools_enabled)
        plan = plan_next_action(
            phase="pre_llm",
            tools_enabled=route.tools_enabled,
        )
        stop = evaluate_loop_stop(
            plan=plan,
            tools_enabled=route.tools_enabled,
        )
        self.assertEqual(plan.action, ACTION_CALL_LLM)
        self.assertEqual(stop.action, STOP_CONTINUE_LLM)
        self.assertNotIn(
            stop.action,
            {
                STOP_FORCE_CONTINUATION,
                STOP_FORCE_TARGET_READ,
                STOP_CONTINUE_NUDGE,
            },
        )


class TestForceFinalRegression(unittest.TestCase):
    """Topic 35 force-final continuation / unread fallback must still hold."""

    def test_deferral_plan_named_force_not_finalize(self):
        continuation = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=True,
            step=1,
            max_steps=8,
            pending_continuation=True,
            total_tool_executions=1,
            tool_budget_total=20,
            uncovered_count=0,
        )
        self.assertEqual(continuation.action, ACTION_FORCE_CONTINUATION)
        self.assertEqual(
            evaluate_loop_stop(plan=continuation).action,
            STOP_FORCE_CONTINUATION,
        )

        unread = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=True,
            step=1,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=1,
            tool_budget_total=20,
            uncovered_count=2,
        )
        self.assertEqual(unread.action, ACTION_FORCE_TARGET_READ)
        self.assertEqual(
            evaluate_loop_stop(plan=unread).action,
            STOP_FORCE_TARGET_READ,
        )

        clear = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=True,
            step=7,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=5,
            tool_budget_total=20,
            uncovered_count=0,
        )
        self.assertEqual(clear.action, ACTION_FINALIZE)
        self.assertEqual(
            evaluate_loop_stop(plan=clear).action,
            STOP_FINALIZE_COMPLETE,
        )

    def test_topic35_failed_force_target_read_still_finalizes(self):
        from tests.test_agent_planning import TestForceFinalDeferralFallback

        TestForceFinalDeferralFallback().test_failed_force_target_read_under_force_final_finalizes()

    def test_topic35_failed_force_continuation_still_finalizes(self):
        from tests.test_agent_planning import TestForceFinalDeferralFallback

        TestForceFinalDeferralFallback().test_failed_force_continuation_under_force_final_tries_unread_then_finalizes()


class TestNoExtraLlmOrRetries(unittest.TestCase):
    def test_evaluate_loop_stop_never_calls_groq(self):
        with patch("app.chatbot.client") as mock_client:
            evaluate_loop_stop(
                plan=PlanDecision(action=ACTION_CALL_LLM, reason="x"),
                tools_enabled=True,
            )
            evaluate_loop_stop(outcome=OUTCOME_REPEATED_FAILURE)
            mock_client.chat.completions.create.assert_not_called()

    def test_stopping_module_has_no_retry_hooks(self):
        import app.services.agent_stopping as stopping
        from pathlib import Path

        source = Path(stopping.__file__).read_text(encoding="utf-8")
        self.assertNotIn("decide_llm_retry", source)
        self.assertNotIn("chat.completions", source)
        self.assertNotIn("time.sleep", source)


class TestChatbotFacadeWiring(unittest.TestCase):
    def test_chatbot_imports_and_calls_evaluate_loop_stop(self):
        import inspect

        src = inspect.getsource(chatbot.chat_with_ai)
        self.assertIn("evaluate_loop_stop", src)
        self.assertIn("OUTCOME_REPEATED_FAILURE", src)
        self.assertIn("OUTCOME_FINALIZE_NONE", src)

    def test_stop_decision_dataclass_shape(self):
        stop = StopDecision(
            action=STOP_CONTINUE_LLM,
            reason="r",
            plan_action=ACTION_CALL_LLM,
        )
        self.assertEqual(stop.action, STOP_CONTINUE_LLM)
        self.assertEqual(stop.plan_action, ACTION_CALL_LLM)


class TestRepeatedFailurePartialBehavior(unittest.TestCase):
    def test_repeated_failure_maps_without_changing_cap(self):
        # Cap remains chatbot.MAX_SAME_TOOL_FAILURES — façade only names it.
        self.assertEqual(chatbot.MAX_SAME_TOOL_FAILURES, 3)
        stop = evaluate_loop_stop(outcome=OUTCOME_REPEATED_FAILURE)
        self.assertEqual(stop.action, STOP_REPEATED_FAILURE)
        # No new predicate: mapping does not inspect failure counts.
        stop2 = evaluate_loop_stop(outcome=OUTCOME_REPEATED_FAILURE)
        self.assertEqual(stop.action, stop2.action)


if __name__ == "__main__":
    unittest.main()
