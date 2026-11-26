"""Topic 37 — deterministic agent reflection."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services import agent_orchestration as orch
from app.services.agent_reflection import (
    QUALITY_BLOCKED,
    QUALITY_EMPTY,
    QUALITY_NESTED_FAILURE,
    QUALITY_OK,
    QUALITY_SPAN_LIMITED,
    REFLECT_ACCEPT,
    REFLECT_FINALIZE_PARTIAL,
    REFLECT_GATHER_MORE,
    REFLECT_REVISE,
    reflect_on_answer,
    reflect_on_tool_result,
)


class TestReflectOnAnswer(unittest.TestCase):
    def test_premature_with_missing_targets_gather_more(self):
        req = {
            "needs_search": True,
            "needs_source_reads": True,
            "min_source_reads": 1,
            "required_targets": [{"id": "execute_tool"}],
            "topics": ["tool_execution"],
        }
        evidence = orch.new_investigation_evidence_state()
        # Search discovered but not verified → uncovered.
        orch.register_search_discoveries(
            evidence,
            req,
            [
                {
                    "file_path": "app/chatbot.py",
                    "line": 100,
                    "text": "def execute_tool(",
                }
            ],
        )
        draft = "Just let me know what you would like to explore."
        result = reflect_on_answer(
            draft=draft,
            requirements=req,
            evidence=evidence,
            evidence_snippets=["search_code: execute_tool hit"],
            tools_enabled=True,
            requirements_met=False,
            can_continue_investigation=True,
            successful_searches=1,
            successful_reads=0,
        )
        self.assertEqual(result.action, REFLECT_GATHER_MORE)
        self.assertTrue(result.premature)
        self.assertIn("execute_tool", result.missing_targets)

    def test_complete_requirements_accept(self):
        req = {
            "needs_search": False,
            "needs_source_reads": False,
            "needs_calculator": True,
            "required_targets": [],
        }
        draft = (
            "The calculator evaluated 2+2 and returned 4 as the arithmetic "
            "result for this request."
        )
        result = reflect_on_answer(
            draft=draft,
            requirements=req,
            evidence={},
            evidence_snippets=["calculator: 4"],
            tools_enabled=True,
            requirements_met=True,
            can_continue_investigation=False,
            successful_searches=0,
            successful_reads=0,
        )
        self.assertIn(result.action, {REFLECT_ACCEPT, REFLECT_REVISE})
        self.assertFalse(result.premature)
        self.assertEqual(result.missing_targets, [])

    def test_revise_reconcile_does_not_mark_verified_missing(self):
        req = {
            "needs_source_reads": True,
            "min_source_reads": 1,
            "required_targets": [{"id": "execute_tool", "feature": "tool_execution"}],
        }
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/chatbot.py",
            line_start=100,
            line_end=140,
            content=(
                "def execute_tool(name, arguments, user_id=None):\n"
                "    return {'success': True}\n"
            ),
            function_complete=True,
        )
        draft = (
            "| Target | Status |\n"
            "| execute_tool | MISSING_SOURCE_EVIDENCE |\n"
            "\n"
            "execute_tool validates arguments then runs the tool."
        )
        result = reflect_on_answer(
            draft=draft,
            requirements=req,
            evidence=evidence,
            evidence_snippets=["read_file: def execute_tool"],
            tools_enabled=True,
            requirements_met=True,
            can_continue_investigation=False,
            successful_searches=1,
            successful_reads=1,
        )
        self.assertEqual(result.action, REFLECT_REVISE)
        self.assertIsNotNone(result.revised_answer)
        revised = result.revised_answer or ""
        # Ledger must not keep verified work labeled missing in the authoritative table.
        self.assertIn("Evidence ledger", revised)
        row = orch.classify_target_evidence_status(
            "execute_tool",
            requirements=req,
            evidence=evidence,
        )
        if row.get("status") == "PASS":
            self.assertNotRegex(
                revised.split("Model synthesis")[0],
                r"execute_tool[^\n]*MISSING_SOURCE_EVIDENCE",
            )

    def test_revise_never_calls_llm(self):
        with patch("app.chatbot.client") as mock_client:
            result = reflect_on_answer(
                draft="Hello there, thanks for asking about the system.",
                requirements={},
                evidence={},
                evidence_snippets=[],
                tools_enabled=False,
                requirements_met=True,
            )
            mock_client.chat.completions.create.assert_not_called()
        self.assertIn(result.action, {REFLECT_ACCEPT, REFLECT_REVISE})

    def test_finalize_partial_when_incomplete_no_continue(self):
        req = {
            "needs_source_reads": True,
            "min_source_reads": 1,
            "required_targets": [{"id": "execute_tool"}],
            "topics": ["tools"],
        }
        draft = "Just let me know."
        result = reflect_on_answer(
            draft=draft,
            requirements=req,
            evidence=orch.new_investigation_evidence_state(),
            evidence_snippets=["search_code: hit"],
            tools_enabled=True,
            requirements_met=False,
            can_continue_investigation=False,
            successful_searches=1,
            successful_reads=0,
        )
        self.assertEqual(result.action, REFLECT_FINALIZE_PARTIAL)
        self.assertTrue(result.premature)

    def test_direct_answer_tools_disabled_unchanged(self):
        result = reflect_on_answer(
            draft="Hi! How can I help you today with your project?",
            requirements={},
            evidence={},
            evidence_snippets=[],
            tools_enabled=False,
            requirements_met=True,
        )
        self.assertIn(result.action, {REFLECT_ACCEPT, REFLECT_REVISE})
        self.assertEqual(result.missing_targets, [])
        self.assertFalse(result.premature)
        self.assertEqual(result.reason, "direct_or_rag_tools_disabled")


class TestReflectOnToolResult(unittest.TestCase):
    def test_empty_tool_result(self):
        result = reflect_on_tool_result(
            tool_name="read_file",
            tool_result={"success": True, "result": "   "},
            blocked=False,
            outer_ok=True,
            nested_ok=True,
            inner_payload={"success": True},
        )
        self.assertEqual(result.tool_quality, QUALITY_EMPTY)

    def test_nested_success_false(self):
        result = reflect_on_tool_result(
            tool_name="read_file",
            tool_result={"success": True, "result": "{}"},
            blocked=False,
            outer_ok=True,
            nested_ok=False,
            inner_payload={"success": False, "error_code": "FILE_NOT_FOUND"},
        )
        self.assertEqual(result.tool_quality, QUALITY_NESTED_FAILURE)

    def test_blocked_duplicate_non_failure(self):
        result = reflect_on_tool_result(
            tool_name="read_file",
            tool_result={"success": True, "blocked": True, "result": "prior"},
            blocked=True,
            error_code="DUPLICATE_TOOL_CALL",
            outer_ok=True,
            nested_ok=True,
        )
        self.assertEqual(result.tool_quality, QUALITY_BLOCKED)

    def test_span_limited(self):
        result = reflect_on_tool_result(
            tool_name="read_file",
            tool_result={"success": True, "result": "body"},
            blocked=False,
            outer_ok=True,
            nested_ok=True,
            inner_payload={"success": True, "span_limited": True, "content": "x"},
        )
        self.assertEqual(result.tool_quality, QUALITY_SPAN_LIMITED)

    def test_ok(self):
        result = reflect_on_tool_result(
            tool_name="search_code",
            tool_result={"success": True, "result": "matches: 1"},
            blocked=False,
            outer_ok=True,
            nested_ok=True,
            inner_payload={"success": True},
        )
        self.assertEqual(result.tool_quality, QUALITY_OK)


class TestReflectionNeverCallsGroq(unittest.TestCase):
    def test_reflect_functions_do_not_touch_groq_client(self):
        with patch("app.chatbot.client") as mock_client:
            reflect_on_answer(
                draft="Answer with enough detail about validation success.",
                requirements={"needs_calculator": True},
                evidence={},
                evidence_snippets=["calculator: 1"],
                tools_enabled=True,
                requirements_met=True,
            )
            reflect_on_tool_result(
                tool_name="calculator",
                tool_result={"success": True, "result": "1"},
                outer_ok=True,
                nested_ok=True,
                inner_payload={"success": True},
            )
            mock_client.chat.completions.create.assert_not_called()


class TestPlanningRegressionUnaffected(unittest.TestCase):
    def test_post_llm_continuation_still_forced(self):
        from app.services.agent_planning import (
            ACTION_FORCE_CONTINUATION,
            plan_next_action,
        )

        decision = plan_next_action(
            phase="post_llm_stop",
            tools_enabled=True,
            step=1,
            max_steps=8,
            pending_continuation=True,
            total_tool_executions=1,
            tool_budget_total=20,
            uncovered_count=0,
            requirements_met=False,
            premature_answer=True,
            has_evidence_snippets=True,
        )
        self.assertEqual(decision.action, ACTION_FORCE_CONTINUATION)

    def test_force_final_still_finalizes_when_clear(self):
        from app.services.agent_planning import (
            ACTION_FINALIZE,
            plan_next_action,
        )

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
        self.assertEqual(decision.reason, "force_final")


if __name__ == "__main__":
    unittest.main()
