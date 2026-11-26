"""Topic 35 — AgentPlanning unit and integration tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services.agent_orchestration import (
    FINALIZATION_STEP_RESERVE,
    derive_investigation_requirements,
)
from app.services.agent_planning import (
    ACTION_CALL_LLM,
    ACTION_CONTINUE_WITH_NUDGE,
    ACTION_FINALIZE,
    ACTION_FORCE_CONTINUATION,
    ACTION_FORCE_TARGET_READ,
    AgentPlan,
    PlanDecision,
    build_agent_plan,
    build_initial_investigation_plan_message,
    plan_next_action,
)


class TestAgentPlanConstruction(unittest.TestCase):
    def test_build_from_requirements(self):
        req = derive_investigation_requirements(
            "Inspect classify_intent. Read the complete implementation."
        )
        plan = build_agent_plan(req)
        self.assertIsInstance(plan, AgentPlan)
        self.assertIs(plan.requirements, req)
        self.assertIn("classify_intent", plan.target_ids)
        self.assertTrue(plan.needs_complete_function or plan.target_ids)

    def test_memory_architecture_flag(self):
        req = {
            "topics": ["memory_architecture"],
            "required_targets": [{"id": "save_memory"}],
            "needs_calculator": False,
        }
        plan = build_agent_plan(req)
        self.assertTrue(plan.memory_architecture)
        self.assertEqual(plan.target_ids, ["save_memory"])


class TestInvestigationPlanMessage(unittest.TestCase):
    def test_message_parity_with_targets(self):
        req = {
            "required_targets": [
                {"id": "classify_intent"},
                {"id": "execute_tool"},
            ]
        }
        text = build_initial_investigation_plan_message(req)
        self.assertTrue(text.startswith("Investigation plan: (1) search_code"))
        self.assertIn("classify_intent", text)
        self.assertIn("execute_tool", text)
        self.assertIn("Required source targets:", text)

    def test_message_without_targets(self):
        text = build_initial_investigation_plan_message({})
        self.assertTrue(text.startswith("Investigation plan: (1) search_code"))
        self.assertNotIn("Required source targets:", text)


class TestPlanNextActionPreLlm(unittest.TestCase):
    def test_tools_disabled_calls_llm(self):
        d = plan_next_action(phase="pre_llm", tools_enabled=False)
        self.assertEqual(d.action, ACTION_CALL_LLM)

    def test_pending_continuation_forced(self):
        d = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=False,
            step=1,
            max_steps=8,
            pending_continuation=True,
            total_tool_executions=1,
            tool_budget_total=14,
            finalization_reserve=FINALIZATION_STEP_RESERVE,
        )
        self.assertEqual(d.action, ACTION_FORCE_CONTINUATION)

    def test_force_final_defers_to_continuation(self):
        d = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=True,
            step=5,
            max_steps=8,
            pending_continuation=True,
            total_tool_executions=2,
            tool_budget_total=14,
        )
        self.assertEqual(d.action, ACTION_FORCE_CONTINUATION)
        self.assertEqual(d.reason, "defer_force_final_continuation")

    def test_force_final_finalizes_when_clear(self):
        d = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=True,
            step=7,
            max_steps=8,
            pending_continuation=False,
            uncovered_count=0,
            total_tool_executions=10,
            tool_budget_total=14,
        )
        self.assertEqual(d.action, ACTION_FINALIZE)
        self.assertEqual(d.reason, "force_final")

    def test_unread_drain_when_investigation_started(self):
        d = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            force_final=False,
            step=2,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=2,
            tool_budget_total=14,
            uncovered_count=3,
            investigation_started=True,
            successful_reads=1,
        )
        self.assertEqual(d.action, ACTION_FORCE_TARGET_READ)
        self.assertGreaterEqual(d.max_drain, 1)


class TestPlanNextActionPostLlm(unittest.TestCase):
    def test_post_pending_continuation(self):
        d = plan_next_action(
            phase="post_llm_stop",
            tools_enabled=True,
            step=1,
            max_steps=8,
            pending_continuation=True,
            total_tool_executions=1,
            tool_budget_total=14,
        )
        self.assertEqual(d.action, ACTION_FORCE_CONTINUATION)

    def test_post_unread_force_target(self):
        d = plan_next_action(
            phase="post_llm_stop",
            tools_enabled=True,
            step=1,
            max_steps=8,
            pending_continuation=False,
            total_tool_executions=2,
            tool_budget_total=14,
            uncovered_count=2,
            successful_searches=1,
            successful_reads=0,
            requirements_met=False,
        )
        self.assertEqual(d.action, ACTION_FORCE_TARGET_READ)

    def test_post_nudge_when_premature(self):
        d = plan_next_action(
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
        self.assertEqual(d.action, ACTION_CONTINUE_WITH_NUDGE)

    def test_post_finalize_incomplete(self):
        d = plan_next_action(
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
        self.assertEqual(d.action, ACTION_FINALIZE)
        self.assertEqual(d.reason, "requirements_incomplete")

    def test_decision_is_plan_decision(self):
        d = plan_next_action(phase="pre_llm", tools_enabled=False)
        self.assertIsInstance(d, PlanDecision)


class TestForceFinalDeferralFallback(unittest.TestCase):
    """HIGH fix: failed force_final deferrals must finalize, not call tools LLM."""

    def _base_patches(self):
        return (
            patch("app.chatbot.client"),
            patch("app.chatbot.save_task_context_to_redis"),
            patch("app.chatbot.restore_task_context_from_redis"),
            patch("app.chatbot.add_to_conversation_memory"),
            patch("app.chatbot.get_conversation_memory", return_value=[]),
            patch(
                "app.chatbot.load_long_term_memory",
                return_value={
                    "memories": [],
                    "conflicts": [],
                    "summaries": [],
                },
            ),
            patch("app.chatbot.try_save_explicit_memory", return_value=False),
            patch("app.chatbot.try_update_explicit_memory", return_value=None),
            patch("app.chatbot.try_delete_explicit_memory", return_value=None),
            patch("app.chatbot.try_inspect_memory_metadata", return_value=None),
        )

    def _finalize_create(self, **kwargs):
        self.assertIsNone(kwargs.get("tools"))
        self.assertNotIn("tool_choice", kwargs)
        msg = MagicMock(
            content="Final answer from evidence ledger.",
            tool_calls=None,
        )
        return MagicMock(
            choices=[MagicMock(message=msg)],
            usage=MagicMock(total_tokens=10),
        )

    def test_failed_force_target_read_under_force_final_finalizes(self):
        from app import chatbot
        from app.services.agent_planning import PlanDecision

        create_calls: list[dict] = []

        def create(**kwargs):
            create_calls.append(kwargs)
            return self._finalize_create(**kwargs)

        plan_decision = PlanDecision(
            action=ACTION_FORCE_TARGET_READ,
            reason="defer_force_final_unread_symbol",
            max_drain=1,
        )

        patches = self._base_patches()
        with patches[0] as mock_client, patches[1], patches[2], patches[3], \
                patches[4], patches[5], patches[6], patches[7], patches[8], \
                patches[9], \
                patch(
                    "app.chatbot.should_force_final_answer",
                    return_value=True,
                ), \
                patch(
                    "app.services.agent_planning.plan_next_action",
                    return_value=plan_decision,
                ), \
                patch(
                    "app.chatbot.next_unread_suggested_read",
                    return_value=None,
                ), \
                patch(
                    "app.services.agent_orchestration._suggestion_window_for_symbol",
                    return_value=None,
                ), \
                patch.object(chatbot, "execute_tool") as exec_tool:
            mock_client.chat.completions.create.side_effect = create
            answer = chatbot.chat_with_ai(
                "Inspect classify_intent. Read the complete implementation.",
                user_id="topic35-force-target-fail",
            )

        self.assertIsInstance(answer, str)
        self.assertTrue(create_calls, "expected a finalization LLM call")
        for kwargs in create_calls:
            self.assertIsNone(kwargs.get("tools"))
            self.assertNotIn("tool_choice", kwargs)
        exec_tool.assert_not_called()

    def test_failed_force_continuation_under_force_final_tries_unread_then_finalizes(
        self,
    ):
        from app import chatbot
        from app.services.agent_planning import PlanDecision

        create_calls: list[dict] = []
        unread_attempts = {"n": 0}

        def create(**kwargs):
            create_calls.append(kwargs)
            return self._finalize_create(**kwargs)

        def tracking_unread(*args, **kwargs):
            unread_attempts["n"] += 1
            return None

        plan_decision = PlanDecision(
            action=ACTION_FORCE_CONTINUATION,
            reason="defer_force_final_continuation",
        )

        patches = self._base_patches()
        with patches[0] as mock_client, patches[1], patches[2], patches[3], \
                patches[4], patches[5], patches[6], patches[7], patches[8], \
                patches[9], \
                patch(
                    "app.chatbot.should_force_final_answer",
                    return_value=True,
                ), \
                patch(
                    "app.services.agent_planning.plan_next_action",
                    return_value=plan_decision,
                ), \
                patch(
                    "app.chatbot.seed_known_symbol_suggestions",
                ) as seed_mock, \
                patch(
                    "app.chatbot.next_unread_suggested_read",
                    side_effect=tracking_unread,
                ), \
                patch(
                    "app.services.agent_orchestration._suggestion_window_for_symbol",
                    return_value=None,
                ), \
                patch.object(chatbot, "execute_tool") as exec_tool:
            mock_client.chat.completions.create.side_effect = create
            # pending_continuation is False, so FORCE_CONTINUATION "fails"
            # immediately and must run the unread fallback then finalize.
            answer = chatbot.chat_with_ai(
                "Inspect classify_intent. Read the complete implementation.",
                user_id="topic35-force-cont-fail",
            )

        self.assertIsInstance(answer, str)
        self.assertGreaterEqual(seed_mock.call_count, 1)
        self.assertGreaterEqual(
            unread_attempts["n"],
            1,
            "expected unread-target fallback after failed continuation",
        )
        self.assertTrue(create_calls)
        for kwargs in create_calls:
            self.assertIsNone(kwargs.get("tools"))
            self.assertNotIn("tool_choice", kwargs)
        exec_tool.assert_not_called()

    def test_non_final_force_target_failure_still_allows_tool_llm(self):
        """force_final=False must not finalize solely because a drain failed."""
        from app import chatbot
        from app.services.agent_planning import PlanDecision

        create_calls: list[dict] = []

        def create(**kwargs):
            create_calls.append(kwargs)
            # First call may include tools; answer without tool_calls.
            msg = MagicMock(
                content="Partial investigation notes.",
                tool_calls=None,
            )
            return MagicMock(
                choices=[MagicMock(message=msg)],
                usage=MagicMock(total_tokens=12),
            )

        def fake_plan(**kwargs):
            if kwargs.get("force_final"):
                return PlanDecision(
                    action=ACTION_FINALIZE,
                    reason="force_final",
                )
            if kwargs.get("phase") == "pre_llm":
                return PlanDecision(
                    action=ACTION_FORCE_TARGET_READ,
                    reason="pre_llm_unread_required",
                    max_drain=1,
                )
            return PlanDecision(
                action=ACTION_FINALIZE,
                reason="answer_ready",
            )

        patches = self._base_patches()
        with patches[0] as mock_client, patches[1], patches[2], patches[3], \
                patches[4], patches[5], patches[6], patches[7], patches[8], \
                patches[9], \
                patch(
                    "app.chatbot.should_force_final_answer",
                    return_value=False,
                ), \
                patch(
                    "app.services.agent_planning.plan_next_action",
                    side_effect=fake_plan,
                ), \
                patch(
                    "app.chatbot.next_unread_suggested_read",
                    return_value=None,
                ), \
                patch(
                    "app.services.agent_orchestration._suggestion_window_for_symbol",
                    return_value=None,
                ), \
                patch.object(
                    chatbot,
                    "execute_tool",
                    return_value={"success": True, "result": "x"},
                ):
            mock_client.chat.completions.create.side_effect = create
            answer = chatbot.chat_with_ai(
                "Inspect classify_intent. Read the complete implementation.",
                user_id="topic35-nonfinal-drain-fail",
            )

        self.assertIsInstance(answer, str)
        self.assertTrue(
            any(kwargs.get("tools") for kwargs in create_calls),
            "non-final path should still reach a tool-enabled LLM call",
        )


class TestPlanningChatIntegration(unittest.TestCase):
    @patch("app.chatbot.client")
    @patch("app.chatbot.save_task_context_to_redis")
    @patch("app.chatbot.restore_task_context_from_redis")
    @patch("app.chatbot.add_to_conversation_memory")
    @patch("app.chatbot.get_conversation_memory", return_value=[])
    @patch("app.chatbot.load_long_term_memory")
    def test_codebase_step0_uses_plan_message(
        self,
        mock_ltm,
        _mock_conv,
        _mock_add,
        _mock_restore,
        _mock_save,
        mock_client,
    ):
        mock_ltm.return_value = {
            "memories": [],
            "conflicts": [],
            "summaries": [],
        }
        captured_kwargs: list = []

        def _fake_create(**kwargs):
            captured_kwargs.append(kwargs)
            msg = MagicMock()
            msg.content = (
                "classify_intent routes educational questions to "
                "technical_question without tools."
            )
            msg.tool_calls = None
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = MagicMock(total_tokens=20)
            return resp

        mock_client.chat.completions.create.side_effect = _fake_create

        from app import chatbot

        # Force early finalization by answering without tools after plan inject.
        answer = chatbot.chat_with_ai(
            "Inspect classify_intent in the codebase. "
            "Read the complete implementation.",
            user_id="test-user-topic35",
        )
        self.assertIsInstance(answer, str)
        # At least one Groq call should have included the investigation plan.
        found_plan = False
        for kwargs in captured_kwargs:
            for m in kwargs.get("messages") or []:
                content = str(m.get("content") or "")
                if content.startswith("Investigation plan: (1) search_code"):
                    found_plan = True
                    break
        self.assertTrue(
            found_plan,
            "Step-0 investigation plan message was not sent to Groq",
        )


if __name__ == "__main__":
    unittest.main()
