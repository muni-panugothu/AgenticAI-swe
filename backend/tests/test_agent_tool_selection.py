"""Topic 39 — typed per-step tool selection."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app import chatbot
from app.services.agent_tool_selection import (
    MODE_AUTO,
    MODE_DISABLED,
    MODE_FINALIZE_NONE,
    MODE_FORCED,
    select_tools_for_step,
)
from app.services.agent_planning import (
    ACTION_FORCE_CONTINUATION,
    ACTION_FORCE_TARGET_READ,
    ACTION_FINALIZE,
    plan_next_action,
)
from app.services.agent_routing import route_request


def _catalog(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in names
    ]


class TestSelectToolsForStep(unittest.TestCase):
    def test_forced_calculator_selection(self):
        defs = _catalog("calculator", "search_code")
        decision = select_tools_for_step(
            tools_enabled=True,
            user_message="Calculate 25 * 40",
            step=0,
            max_steps=8,
            calculator_used=False,
            active_definitions=defs,
            resolve_tool_choice_fn=chatbot.resolve_tool_choice,
        )
        self.assertEqual(decision.mode, MODE_FORCED)
        self.assertEqual(
            decision.tool_choice,
            {"type": "function", "function": {"name": "calculator"}},
        )
        self.assertEqual(decision.tool_definitions, defs)
        self.assertIsNotNone(decision.nudge)
        self.assertIn("calculator", (decision.nudge or "").lower())

    def test_multi_tool_automatic_selection(self):
        msg = "Search the code for calculator and also calculate 2+2"
        self.assertTrue(chatbot.is_multi_tool_request(msg))
        defs = list(chatbot.tool_definitions)
        decision = select_tools_for_step(
            tools_enabled=True,
            user_message=msg,
            step=0,
            max_steps=8,
            calculator_used=False,
            active_definitions=defs,
            resolve_tool_choice_fn=chatbot.resolve_tool_choice,
        )
        self.assertEqual(decision.mode, MODE_AUTO)
        self.assertEqual(decision.tool_choice, "auto")
        self.assertEqual(decision.tool_definitions, defs)
        self.assertIsNotNone(decision.nudge)
        self.assertIn("multiple tools", (decision.nudge or "").lower())

    def test_tools_disabled_selection(self):
        decision = select_tools_for_step(
            tools_enabled=False,
            user_message="Hi there",
            step=0,
            max_steps=8,
            calculator_used=False,
            active_definitions=list(chatbot.tool_definitions),
            resolve_tool_choice_fn=chatbot.resolve_tool_choice,
        )
        self.assertEqual(decision.mode, MODE_DISABLED)
        self.assertIsNone(decision.tool_choice)
        self.assertEqual(decision.tool_definitions, [])
        self.assertIsNone(decision.nudge)
        self.assertEqual(decision.reason, "tools_disabled")

    def test_final_step_no_tools(self):
        defs = _catalog("search_code", "read_file")
        decision = select_tools_for_step(
            tools_enabled=True,
            user_message="Find execute_tool",
            step=7,
            max_steps=8,
            calculator_used=False,
            active_definitions=defs,
            resolve_tool_choice_fn=chatbot.resolve_tool_choice,
        )
        self.assertEqual(decision.mode, MODE_FINALIZE_NONE)
        self.assertEqual(decision.tool_choice, "none")
        self.assertEqual(decision.tool_definitions, [])
        self.assertIsNotNone(decision.nudge)

    def test_active_catalog_preservation(self):
        defs = _catalog("search_code", "read_file")
        decision = select_tools_for_step(
            tools_enabled=True,
            user_message="Find my agent loop implementation",
            step=1,
            max_steps=8,
            calculator_used=False,
            active_definitions=defs,
            resolve_tool_choice_fn=chatbot.resolve_tool_choice,
        )
        self.assertEqual(decision.mode, MODE_AUTO)
        self.assertEqual(decision.tool_definitions, defs)
        names = {
            (t.get("function") or {}).get("name") for t in decision.tool_definitions
        }
        self.assertEqual(names, {"search_code", "read_file"})

    def test_rag_tools_off_via_routing_catalog(self):
        route = route_request(
            user_message="What does the document say?",
            rag_context="Evidence about RAG.",
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertFalse(route.tools_enabled)
        decision = select_tools_for_step(
            tools_enabled=route.tools_enabled,
            user_message="What does the document say?",
            step=0,
            max_steps=8,
            calculator_used=False,
            active_definitions=route.tool_definitions,
            resolve_tool_choice_fn=chatbot.resolve_tool_choice,
        )
        self.assertEqual(decision.mode, MODE_DISABLED)
        self.assertEqual(decision.tool_definitions, [])

    def test_selection_service_never_calls_llm(self):
        with patch("app.chatbot.client") as mock_client:
            select_tools_for_step(
                tools_enabled=True,
                user_message="Calculate 1+1",
                step=0,
                max_steps=8,
                calculator_used=False,
                active_definitions=_catalog("calculator"),
                resolve_tool_choice_fn=chatbot.resolve_tool_choice,
            )
            mock_client.chat.completions.create.assert_not_called()


class TestPlanningForceBypass(unittest.TestCase):
    def test_force_continuation_still_preferred_over_llm_selection(self):
        decision = plan_next_action(
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
        self.assertEqual(decision.action, ACTION_FORCE_CONTINUATION)

    def test_force_target_read_still_preferred(self):
        decision = plan_next_action(
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
        self.assertEqual(decision.action, ACTION_FORCE_TARGET_READ)

    def test_force_final_finalize_when_clear(self):
        decision = plan_next_action(
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
        self.assertEqual(decision.action, ACTION_FINALIZE)


class TestChatIntegrationSelection(unittest.TestCase):
    def test_calculator_path_uses_forced_tool_choice(self):
        captured = {}

        def create(**kwargs):
            # First LLM turn must carry forced calculator selection; later
            # finalization calls are tool-free by design.
            if "first_tool_choice" not in captured:
                captured["first_tools"] = kwargs.get("tools")
                captured["first_tool_choice"] = kwargs.get("tool_choice")
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="42",
                            tool_calls=None,
                        )
                    )
                ],
                usage=MagicMock(total_tokens=10),
            )

        with patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot,
            "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(chatbot, "restore_task_context_from_redis"), patch.object(
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
            chatbot._get_conversation_store("sel-calc").clear()
            answer = chatbot.chat_with_ai(
                "Calculate 6 * 7",
                user_id="sel-calc",
                request_id="req-sel-calc",
            )

        self.assertTrue(captured.get("first_tools"))
        self.assertEqual(
            captured.get("first_tool_choice"),
            {"type": "function", "function": {"name": "calculator"}},
        )
        self.assertIn("42", answer)


if __name__ == "__main__":
    unittest.main()
